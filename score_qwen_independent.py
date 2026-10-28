import json
import os

from retrieval_harm import load_cases
from revision_evals import evaluate, library
from score_qwen import claims_of

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "ptbxl_baseline", "open_models", "Qwen__Qwen3-8B.jsonl")
REPORT = os.path.join(ROOT, "results", "ptbxl_baseline", "qwen_independent.json")


def policies(record, facts, facts_unig, checks):
    from claim_scoring import filter_note

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
            no_rag = claims_of(item.get("no_rag", ""))
            keyword = claims_of(item.get("with_keyword", ""))
            tfidf = claims_of(item.get("with_tfidf", ""))
            if no_rag is None or keyword is None or tfidf is None:
                continue
            notes[int(item["ecg_id"])] = {
                "no_rag": no_rag,
                "with_keyword": keyword,
                "with_tfidf": tfidf,
            }
    report = {"model": "Qwen/Qwen3-8B", **evaluate(notes, load_cases(), library(), policies)}
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    arms = report["arms"]
    print(json.dumps({
        "scored": report["scored_notes"],
        "withdrawals": report["withdrawals_by_12sl_check"],
        "keyword": {
            "dx_expert": arms["keyword"]["diagnosis_absent_from_expert_label"],
            "crit_12": arms["keyword"]["criterion_rejected_by_12sl_or_unchecked"],
            "crit_unig": arms["keyword"]["criterion_rejected_by_unig"],
            "supported": arms["keyword"]["mean_applied_supported_by_12sl"],
            "unchecked": arms["keyword"]["mean_applied_without_check"],
            "rejected": arms["keyword"]["mean_applied_rejected_by_12sl"],
        },
        "check_both": {
            "crit_unig": arms["check_both"]["criterion_rejected_by_unig"],
            "supported": arms["check_both"]["mean_applied_supported_by_12sl"],
            "statement": arms["check_both"]["statement_mismatch"],
        },
        "no_retrieval_dx": arms["no_retrieval"]["diagnosis_absent_from_expert_label"],
    }, indent=2))


if __name__ == "__main__":
    main()
