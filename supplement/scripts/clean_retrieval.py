import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
\
import pandas as pd
\
from claim_scoring import filter_note
from heldout_notes import (
    complete,
    fit_tfidf,
    keyword_retrieve,
    library,
    load_jsonl,
    note_prompt,
    tfidf_retrieve,
)
from retrieval_harm import facts_for, load_cases, load_env, log
from revision_evals import evaluate
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "clean_generations.jsonl")
FAILS = os.path.join(OUT, "clean_failures.jsonl")
PILOT_RAW = os.path.join(OUT, "clean_pilot.jsonl")
REPORT = os.path.join(OUT, "clean_retrieval.json")
SAMPLE = os.path.join(OUT, "heldout_sample.csv")
WORKERS = 4
PILOT_N = 8
\
\
def numeric_cards(cards: list[dict]) -> list[dict]:
    kept = [card for card in cards if card.get("check")]
    if len(kept) != 14:
        raise SystemExit(f"expected 14 numeric cards, found {len(kept)}")
    return kept
def generate_row(row: pd.Series, cards: list[dict], vectorizer, matrix) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
    with_keyword = complete(note_prompt(facts, keyword))
    with_tfidf = complete(note_prompt(facts, tfidf))
    return {
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "keyword_cards": [card["id"] for card in keyword],
        "tfidf_cards": [card["id"] for card in tfidf],
        "with_keyword": with_keyword["claims"],
        "with_tfidf": with_tfidf["claims"],
        "models": [with_keyword["model"], with_tfidf["model"]],
        "finish_reasons": [with_keyword["finish_reason"], with_tfidf["finish_reason"]],
        "usage": [with_keyword["usage"], with_tfidf["usage"]],
    }
def run_generation(sample: pd.DataFrame, raw_path: str, fail_path: str, cards: list[dict], vectorizer, matrix) -> None:
    done = load_jsonl(raw_path)
    failed = load_jsonl(fail_path) if fail_path and os.path.exists(fail_path) else {}
    pending = [row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed]
    log(f"clean sample {len(sample)} pending {len(pending)}")
    lock = threading.Lock()
    \
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
        log(f"clean done {ecg_id}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_row, row, cards, vectorizer, matrix): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            finish(futures[future], future)
def policies(record, facts, facts_unig, checks):
    return {
        "no_retrieval": record.get("no_rag", []),
        "keyword": record["with_keyword"],
        "tfidf": record["with_tfidf"],
        "check_12sl": filter_note(record["with_keyword"], facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(record["with_keyword"], facts, facts_unig, checks, "agree"),
    }
def attach_existing_notes(clean: dict[int, dict]) -> dict[int, dict]:
    original = load_jsonl(os.path.join(OUT, "heldout_generations.jsonl"))
    merged = {}
    for ecg_id, record in clean.items():
        if ecg_id not in original:
            continue
        merged[ecg_id] = {**record, "no_rag": original[ecg_id]["no_rag"]}
    return merged
def score(cases: pd.DataFrame, cards: list[dict]) -> None:
    notes = attach_existing_notes(load_jsonl(RAW))
    report = evaluate(notes, cases, cards, policies)
    keyword_sets = []
    distractors = 0
    slots = 0
    for record in load_jsonl(RAW).values():
        keyword_sets.append(tuple(record["keyword_cards"]))
        slots += len(record["keyword_cards"])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "notes": 0}
    for record in load_jsonl(RAW).values():
        for item in record.get("usage") or []:
            if not item:
                continue
            usage["prompt_tokens"] += int(item.get("prompt_tokens") or 0)
            usage["completion_tokens"] += int(item.get("completion_tokens") or 0)
            usage["notes"] += 1
    frame_keyword = report["arms"]["keyword"]["criterion_rejected_by_12sl_or_unchecked"]
    frame_check = report["arms"]["check_both"]["criterion_rejected_by_12sl_or_unchecked"]
    report = {
        "library": "14 numeric cards from ecg_criteria_v2.json; cards with a null check were removed",
        "distractor_slots_in_keyword_prompts": distractors,
        "keyword_slots": slots,
        "keyword_unique_top3_sets": int(len(set(keyword_sets))),
        "usage": usage,
        "circular_criterion_case_rate": {
            "keyword": frame_keyword,
            "check_both": frame_check,
        },
        **report,
    }
    by_id = {int(row.ecg_id): row for row in pd.read_csv(SAMPLE).itertuples(index=False)}
    \
\
    del by_id
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "scored": report["scored_notes"],
        "arms": {
            name: {
                "criterion_12sl": report["arms"][name]["criterion_rejected_by_12sl_or_unchecked"],
                "criterion_unig": report["arms"][name]["criterion_rejected_by_unig"],
                "diagnosis_expert": report["arms"][name]["diagnosis_absent_from_expert_label"],
                "mean_supported_12sl": report["arms"][name]["mean_applied_supported_by_12sl"],
                "mean_rejected_12sl": report["arms"][name]["mean_applied_rejected_by_12sl"],
                "mean_unchecked": report["arms"][name]["mean_applied_without_check"],
            }
            for name in report["arms"]
        },
        "contrasts": report["contrasts"],
        "usage": usage,
    }, indent=2))
def pilot(cases: pd.DataFrame, cards: list[dict], vectorizer, matrix) -> None:
    pool = cases[cases["split"] == "calibrate"]
    chosen = pool.sample(n=PILOT_N, random_state=19)
    if os.path.exists(PILOT_RAW):
        done = load_jsonl(PILOT_RAW)
    else:
        done = {}
    if len(done) < PILOT_N:
        run_generation(chosen, PILOT_RAW, "", cards, vectorizer, matrix)
        done = load_jsonl(PILOT_RAW)
    reasons = []
    for note in done.values():
        reasons.extend(note["finish_reasons"])
        if not note["with_keyword"] or not note["with_tfidf"]:
            raise SystemExit("pilot note was empty")
    passed = len(done) >= PILOT_N and reasons.count("stop") == len(reasons) and len(reasons) > 0
    log(json.dumps({"clean_pilot_passed": passed, "notes": len(done), "reasons": reasons.count("stop")}, indent=2))
    if not passed:
        raise SystemExit(1)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "run", "score"])
    args = parser.parse_args()
    load_env()
    cards = numeric_cards(library())
    cases = load_cases()
    vectorizer, matrix = fit_tfidf(cards)
    if args.command == "pilot":
        pilot(cases, cards, vectorizer, matrix)
    elif args.command == "run":
        pilot(cases, cards, vectorizer, matrix)
        chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
        sample = cases[cases["ecg_id"].isin(chosen)].copy()
        run_generation(sample, RAW, FAILS, cards, vectorizer, matrix)
        score(cases, cards)
    else:
        score(cases, cards)
if __name__ == "__main__":
    main()
