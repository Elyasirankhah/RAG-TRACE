import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from claim_scoring import filter_note
from heldout_notes import keyword_retrieve, library, load_jsonl, note_prompt
from retrieval_harm import claim_list, facts_for, load_cases, load_env, log, parse_json_content
from revision_evals import evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "constrained_generations.jsonl")
FAILS = os.path.join(OUT, "constrained_failures.jsonl")
PILOT_RAW = os.path.join(OUT, "constrained_pilot.jsonl")
REPORT = os.path.join(OUT, "constrained_results.json")
SAMPLE = os.path.join(OUT, "heldout_sample.csv")
WORKERS = 4
PILOT_N = 8


def numeric_cards(cards: list[dict]) -> list[dict]:
    kept = [card for card in cards if card.get("check")]
    if len(kept) != 14:
        raise SystemExit(f"expected 14 numeric cards, found {len(kept)}")
    return kept


def schema_for(card_ids: list[str]) -> dict:
    branches = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["diagnosis"]},
                "text": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["type", "text", "label"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["measurement"]},
                "text": {"type": "string"},
                "name": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["type", "text", "name", "value"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["limit"]},
                "text": {"type": "string"},
            },
            "required": ["type", "text"],
        },
    ]
    if card_ids:
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["criterion"]},
                    "text": {"type": "string"},
                    "card_id": {"type": "string", "enum": card_ids},
                    "applies": {"type": "boolean"},
                },
                "required": ["type", "text", "card_id", "applies"],
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"claims": {"type": "array", "maxItems": 8, "items": {"anyOf": branches}}},
        "required": ["claims"],
    }


def complete_constrained(messages: list[dict], card_ids: list[str]) -> dict:
    load_env()
    base = os.environ["API_BASE_URL"].rstrip("/")
    body = json.dumps(
        {
            "model": os.environ["API_MODEL"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ecg_note", "strict": True, "schema": schema_for(card_ids)},
            },
        }
    ).encode()
    last_error = None
    for attempt in range(1, 5):
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            parsed = parse_json_content(content)
            return {
                "claims": claim_list(parsed),
                "model": payload.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"),
            }
        except Exception as exc:
            last_error = exc
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:400]
            log(f"retry {attempt}: {type(exc).__name__} {detail}")
            time.sleep(2 * attempt)
    raise RuntimeError(last_error)


def generate_row(row: pd.Series, cards: list[dict]) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    ids = [card["id"] for card in keyword]
    note = complete_constrained(note_prompt(facts, keyword), ids)
    return {
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "keyword_cards": ids,
        "with_keyword": note["claims"],
        "models": [note["model"]],
        "finish_reasons": [note["finish_reason"]],
        "usage": [note["usage"]],
    }


def run_generation(sample: pd.DataFrame, raw_path: str, fail_path: str, cards: list[dict]) -> None:
    done = load_jsonl(raw_path)
    failed = load_jsonl(fail_path) if fail_path and os.path.exists(fail_path) else {}
    pending = [row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed]
    log(f"constrained sample {len(sample)} pending {len(pending)}")
    lock = threading.Lock()

    def finish(ecg_id: int, future) -> None:
        try:
            item = future.result()
        except Exception as exc:
            item = {"ecg_id": ecg_id, "error": f"{type(exc).__name__}: {exc}"}
            if fail_path:
                with lock, open(fail_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
            log(f"FAIL ecg {ecg_id}: {item['error']}")
            return
        with lock, open(raw_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
        log(f"constrained done {ecg_id}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_row, row, cards): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            finish(futures[future], future)


def policies(record, facts, facts_unig, checks):
    original = record.get("no_rag") or []
    return {
        "no_retrieval": original,
        "keyword": record["with_keyword"],
        "check_12sl": filter_note(record["with_keyword"], facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(record["with_keyword"], facts, facts_unig, checks, "agree"),
    }


def attach_original(notes: dict) -> dict:
    original = load_jsonl(os.path.join(OUT, "heldout_generations.jsonl"))
    merged = {}
    for ecg_id, record in notes.items():
        if ecg_id not in original:
            continue
        merged[ecg_id] = {**record, "no_rag": original[ecg_id]["no_rag"]}
    return merged


def invalid_ids(notes: dict, allowed_of) -> dict:
    empty = invented = outside = applied = 0
    for record in notes.values():
        allowed = set(allowed_of(record))
        for claim in record["with_keyword"]:
            if claim.get("type") != "criterion" or not claim.get("applies"):
                continue
            applied += 1
            card_id = claim.get("card_id")
            if card_id in (None, ""):
                empty += 1
            elif card_id not in allowed:
                invented += 1
                outside += 1
    return {"applied": applied, "empty_id": empty, "invalid_id": invented}


def score(cases: pd.DataFrame, cards: list[dict]) -> None:
    notes = attach_original(load_jsonl(RAW))
    report = evaluate(notes, cases, cards, policies)
    report["constraint"] = "json_schema strict. Criterion card_id must be one of the retrieved numeric-card ids."
    report["identifiers"] = invalid_ids(notes, lambda record: record["keyword_cards"])
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    arms = report["arms"]
    print(json.dumps({
        "scored": report["scored_notes"],
        "identifiers": report["identifiers"],
        "keyword_release": arms["keyword"]["criterion_rejected_by_12sl_or_unchecked"],
        "keyword_unig": arms["keyword"]["criterion_rejected_by_unig"],
        "keyword_expert": arms["keyword"]["diagnosis_absent_from_expert_label"],
        "keyword_unchecked": arms["keyword"]["mean_applied_without_check"],
        "keyword_rejected": arms["keyword"]["mean_applied_rejected_by_12sl"],
        "keyword_supported": arms["keyword"]["mean_applied_supported_by_12sl"],
        "after_release": arms["check_12sl"]["criterion_rejected_by_12sl_or_unchecked"],
        "after_unig": arms["check_12sl"]["criterion_rejected_by_unig"],
        "after_expert": arms["check_12sl"]["diagnosis_absent_from_expert_label"],
        "after_supported": arms["check_12sl"]["mean_applied_supported_by_12sl"],
    }, indent=2))


def pilot(cases: pd.DataFrame, cards: list[dict]) -> None:
    chosen = cases[cases["split"] == "calibrate"].sample(n=PILOT_N, random_state=29)
    if not os.path.exists(PILOT_RAW) or len(load_jsonl(PILOT_RAW)) < PILOT_N:
        run_generation(chosen, PILOT_RAW, "", cards)
    done = load_jsonl(PILOT_RAW)
    reasons = [reason for note in done.values() for reason in note["finish_reasons"]]
    passed = len(done) >= PILOT_N and reasons.count("stop") == len(reasons) and len(reasons) > 0
    bad = invalid_ids(done, lambda record: record["keyword_cards"])
    log(json.dumps({"constrained_pilot_passed": passed, "notes": len(done), "identifiers": bad}, indent=2))
    if not passed or bad["empty_id"] or bad["invalid_id"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "run", "score"])
    args = parser.parse_args()
    load_env()
    cards = numeric_cards(library())
    cases = load_cases()
    if args.command == "pilot":
        pilot(cases, cards)
    elif args.command == "run":
        pilot(cases, cards)
        chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
        sample = cases[cases["ecg_id"].isin(chosen)].copy()
        run_generation(sample, RAW, FAILS, cards)
        score(cases, cards)
    else:
        score(cases, cards)


if __name__ == "__main__":
    main()
