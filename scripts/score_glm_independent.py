import json
import os

from retrieval_harm import load_cases
from revision_evals import evaluate, library
from claim_scoring import filter_note

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "ptbxl_baseline", "glm", "heldout_generations.jsonl")
REPORT = os.path.join(ROOT, "results", "ptbxl_baseline", "glm", "heldout_independent.json")


def policies(record, facts, facts_unig, checks):
    keyword = record["with_keyword"]
    return {
        "no_retrieval": record["no_rag"],
        "keyword": keyword,
        "tfidf": record["with_tfidf"],
        "check_12sl": filter_note(keyword, facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(keyword, facts, facts_unig, checks, "agree"),
    }


def main() -> None:
    notes = {}
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            notes[int(item["ecg_id"])] = {
                "no_rag": item["no_rag"],
                "with_keyword": item["with_keyword"],
                "with_tfidf": item["with_tfidf"],
            }
    report = {"model": "GLM-5.3-Flash", **evaluate(notes, load_cases(), library(), policies)}
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    keyword = report["arms"]["keyword"]
    after = report["arms"]["check_12sl"]
    print(json.dumps({
        "scored": report["scored_notes"],
        "keyword_expert": keyword["diagnosis_absent_from_expert_label"],
        "keyword_unig": keyword["criterion_rejected_by_unig"],
        "keyword_release": keyword["criterion_rejected_by_12sl_or_unchecked"],
        "after_expert": after["diagnosis_absent_from_expert_label"],
        "after_unig": after["criterion_rejected_by_unig"],
        "after_release": after["criterion_rejected_by_12sl_or_unchecked"],
    }, indent=2))


if __name__ == "__main__":
    main()
