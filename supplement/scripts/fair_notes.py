import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
\
import pandas as pd
\
from claim_scoring import filter_note
from heldout_notes import complete, keyword_retrieve, library, load_jsonl
from retrieval_harm import facts_for, load_cases, load_env, log
from revision_evals import evaluate
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "fair_generations.jsonl")
FAILS = os.path.join(OUT, "fair_failures.jsonl")
PILOT_RAW = os.path.join(OUT, "fair_pilot.jsonl")
REPORT = os.path.join(OUT, "fair_results.json")
SAMPLE = os.path.join(OUT, "heldout_sample.csv")
WORKERS = 4
PILOT_N = 8
SYSTEM = (
    "Return one JSON object with a claims list of at most 8 claims. "
    "Each claim has type, text, and optional label, name, value, card_id, applies. "
    "Use only the supplied measurement numbers. Do not invent values. "
    "A diagnosis claim may use only a label whose probability is at least 0.5. "
    "Include one limit claim stating that this is not a clinical diagnosis. "
    "Allowed types: diagnosis, measurement, criterion, limit. "
    "A criterion claim sets card_id and applies to true or false."
)
\
\
def matched_prompt(facts: dict, retrieved: list[dict] | None) -> list[dict]:
    payload = dict(facts)
    if retrieved is not None:
        payload["retrieved_cards"] = [{"id": card["id"], "text": card["text"]} for card in retrieved]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload)},
    ]
def generate_row(row: pd.Series, cards: list[dict]) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    no_rag = complete(matched_prompt(facts, None))
    with_keyword = complete(matched_prompt(facts, keyword))
    return {
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "keyword_cards": [card["id"] for card in keyword],
        "asserted_from_probability": facts["asserted_labels"],
        "no_rag": no_rag["claims"],
        "with_keyword": with_keyword["claims"],
        "models": [no_rag["model"], with_keyword["model"]],
        "finish_reasons": [no_rag["finish_reason"], with_keyword["finish_reason"]],
        "usage": [no_rag["usage"], with_keyword["usage"]],
    }
def run_generation(sample: pd.DataFrame, raw_path: str, fail_path: str, cards: list[dict]) -> None:
    done = load_jsonl(raw_path)
    failed = load_jsonl(fail_path) if fail_path and os.path.exists(fail_path) else {}
    pending = [row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed]
    log(f"fair sample {len(sample)} pending {len(pending)}")
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
        log(f"fair done {ecg_id}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_row, row, cards): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            finish(futures[future], future)
def policies(record, facts, facts_unig, checks):
    return {
        "no_retrieval": record["no_rag"],
        "keyword": record["with_keyword"],
        "check_12sl": filter_note(record["with_keyword"], facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(record["with_keyword"], facts, facts_unig, checks, "agree"),
    }
def score(cases: pd.DataFrame, cards: list[dict]) -> None:
    notes = load_jsonl(RAW)
    report = evaluate(notes, cases, cards, policies)
    report["prompt"] = "Identical instructions. Keyword arm adds retrieved_cards. No arm forbids criterion claims."
    report["retrieval_query"] = "Words of superclasses whose calibrated probability is at least 0.5. Stored test labels are not in the query or the prompt."
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    arms = report["arms"]
    print(json.dumps({
        "scored": report["scored_notes"],
        "no_retrieval": {
            "release": arms["no_retrieval"]["criterion_rejected_by_12sl_or_unchecked"],
            "expert": arms["no_retrieval"]["diagnosis_absent_from_expert_label"],
            "mean_applied": arms["no_retrieval"]["mean_applied_criteria"],
            "mean_unchecked": arms["no_retrieval"]["mean_applied_without_check"],
        },
        "keyword": {
            "release": arms["keyword"]["criterion_rejected_by_12sl_or_unchecked"],
            "expert": arms["keyword"]["diagnosis_absent_from_expert_label"],
            "unig": arms["keyword"]["criterion_rejected_by_unig"],
            "mean_supported": arms["keyword"]["mean_applied_supported_by_12sl"],
        },
        "check_12sl": {
            "release": arms["check_12sl"]["criterion_rejected_by_12sl_or_unchecked"],
            "expert": arms["check_12sl"]["diagnosis_absent_from_expert_label"],
            "unig": arms["check_12sl"]["criterion_rejected_by_unig"],
            "mean_supported": arms["check_12sl"]["mean_applied_supported_by_12sl"],
        },
        "contrasts": report["contrasts"],
    }, indent=2))
def pilot(cases: pd.DataFrame, cards: list[dict]) -> None:
    chosen = cases[cases["split"] == "calibrate"].sample(n=PILOT_N, random_state=23)
    if not os.path.exists(PILOT_RAW) or len(load_jsonl(PILOT_RAW)) < PILOT_N:
        run_generation(chosen, PILOT_RAW, "", cards)
    done = load_jsonl(PILOT_RAW)
    reasons = []
    for note in done.values():
        reasons.extend(note["finish_reasons"])
        if not isinstance(note["no_rag"], list) or not isinstance(note["with_keyword"], list):
            raise SystemExit("pilot note was empty")
    passed = len(done) >= PILOT_N and reasons.count("stop") == len(reasons) and len(reasons) > 0
    log(json.dumps({"fair_pilot_passed": passed, "notes": len(done)}, indent=2))
    if not passed:
        raise SystemExit(1)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "run", "score"])
    args = parser.parse_args()
    load_env()
    cards = library()
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
