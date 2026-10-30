import argparse
import json
import os
import threading
\
import pandas as pd
\
from constrained_notes import complete_constrained, numeric_cards, schema_for
from heldout_notes import keyword_retrieve, library, load_jsonl
from retrieval_harm import facts_for, load_cases, load_env, log
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "matched_constrained_generations.jsonl")
FAILS = os.path.join(OUT, "matched_constrained_failures.jsonl")
PILOT_RAW = os.path.join(OUT, "matched_constrained_pilot.jsonl")
SAMPLE = os.path.join(OUT, "heldout_sample.csv")
WORKERS = 4
PILOT_N = 8
SYSTEM = (
    "Return one JSON object with a claims list of at most 8 claims. "
    "Use only the supplied measurement numbers. Do not invent values. "
    "A diagnosis claim may use only a label whose probability is at least 0.5. "
    "Include one limit claim stating that this is not a clinical diagnosis. "
    "Allowed types are diagnosis, measurement, criterion, and limit. "
    "Emit a criterion claim only when retrieved_cards is present. "
    "Then card_id must be one of those ids and applies is true or false."
)
\
\
def matched_prompt(facts: dict, retrieved: list[dict] | None) -> list[dict]:
    payload = dict(facts)
    if retrieved:
        payload["retrieved_cards"] = [{"id": card["id"], "text": card["text"]} for card in retrieved]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload)},
    ]
def generate_row(row: pd.Series, cards: list[dict]) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    ids = [card["id"] for card in keyword]
    no_rag = complete_constrained(matched_prompt(facts, None), [])
    with_keyword = complete_constrained(matched_prompt(facts, keyword), ids)
    return {
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "keyword_cards": ids,
        "no_rag": no_rag["claims"],
        "with_keyword": with_keyword["claims"],
        "models": [no_rag["model"], with_keyword["model"]],
        "finish_reasons": [no_rag["finish_reason"], with_keyword["finish_reason"]],
        "prompt": "identical system text; user message adds retrieved_cards; schema limits card_id to those ids",
    }
def run_generation(sample: pd.DataFrame, raw_path: str, fail_path: str, cards: list[dict]) -> None:
    done = load_jsonl(raw_path)
    failed = load_jsonl(fail_path) if fail_path and os.path.exists(fail_path) else {}
    pending = [row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed]
    log(f"matched constrained sample {len(sample)} pending {len(pending)}")
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
        log(f"matched constrained done {ecg_id}")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    \
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_row, row, cards): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            finish(futures[future], future)
def pilot(cases: pd.DataFrame, cards: list[dict]) -> None:
    chosen = cases[cases["split"] == "calibrate"].sample(n=PILOT_N, random_state=31)
    if len(load_jsonl(PILOT_RAW)) < PILOT_N:
        run_generation(chosen, PILOT_RAW, "", cards)
    done = load_jsonl(PILOT_RAW)
    reasons = [reason for note in done.values() for reason in note.get("finish_reasons", [])]
    empty_schema = schema_for([])
    passed = len(done) >= PILOT_N and reasons.count("stop") == len(reasons) and "criterion" not in json.dumps(empty_schema)
    log(json.dumps({"matched_pilot_passed": passed, "notes": len(done)}, indent=2))
    if not passed:
        raise SystemExit(1)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pilot", "run"])
    args = parser.parse_args()
    load_env()
    cards = numeric_cards(library())
    cases = load_cases()
    if args.command == "pilot":
        pilot(cases, cards)
        return
    pilot(cases, cards)
    chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
    sample = cases[cases["ecg_id"].isin(chosen)].copy()
    run_generation(sample, RAW, FAILS, cards)
if __name__ == "__main__":
    main()
