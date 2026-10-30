import json
import os
\
from retention_eval import OUT, load_closed, tally
from retrieval_harm import load_cases
from revision_evals import library
from score_qwen import claims_of
\
REPORT = os.path.join(OUT, "retention_paired.json")
\
\
def load_qwen_paired() -> dict:
    notes = {}
    with open(os.path.join(OUT, "open_models", "Qwen__Qwen3-8B.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            arms = [claims_of(item.get(key, "")) for key in ("no_rag", "with_keyword", "with_tfidf")]
            if any(arm is None for arm in arms):
                continue
            notes[int(item["ecg_id"])] = {"with_keyword": arms[1]}
    return notes
def main() -> None:
    cards = library()
    cases = load_cases()
    qwen = load_qwen_paired()
    report = {
        "rule": "12SL withdrawal. Uni-G is the judge and is not an input to this edit.",
        "population": "patients whose no-card, keyword, and TF-IDF notes all parse",
        "gpt-4.1-mini": {"patients": 1083, **tally(load_closed(), cases, cards, lambda record: record["with_keyword"])},
        "Qwen3-8B": {"patients": len(qwen), **tally(qwen, cases, cards, lambda record: record["with_keyword"])},
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
