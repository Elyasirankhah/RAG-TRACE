import json
import os
\
from constrained_notes import invalid_ids, numeric_cards, policies
from diagnosis_prf import score_notes
from exploratory_revision import score_records
from heldout_notes import library, load_jsonl
from retrieval_harm import load_cases
from revision_evals import evaluate
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "matched_constrained_generations.jsonl")
REPORT = os.path.join(OUT, "matched_constrained_results.json")
\
\
def main() -> None:
    cases = load_cases()
    cards = numeric_cards(library())
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    notes = load_jsonl(RAW)
    report = evaluate(notes, cases, cards, policies)
    report["design"] = (
        "Same gpt-4.1-mini, same system text, same strict JSON schema family, temperature 0. "
        "The keyword arm adds retrieved_cards and allows a criterion claim whose card_id must be a retrieved id."
    )
    report["notes_in_file"] = len(notes)
    report["identifiers"] = invalid_ids(notes, lambda record: record["keyword_cards"])
    report["label_aware_false_diagnosis"] = score_records(notes, by_id)
    report["label_aware_completeness"] = score_notes(notes, by_id)
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    arms = report["arms"]
    complete = report["label_aware_completeness"]
    brief = {
        "notes": len(notes),
        "scored": report["scored_notes"],
        "identifiers": report["identifiers"],
        "release": {name: arms[name]["criterion_rejected_by_12sl_or_unchecked"] for name in arms},
        "unig": {name: arms[name]["criterion_rejected_by_unig"] for name in arms},
        "cue_expert": {name: arms[name]["diagnosis_absent_from_expert_label"] for name in arms},
        "supported": {name: arms[name]["mean_applied_supported_by_12sl"] for name in arms},
        "false_dx": report["label_aware_false_diagnosis"].get("keyword_minus_no_retrieval"),
        "no_card": {k: complete["no_retrieval"][k] for k in ("micro_precision", "micro_recall", "micro_f1", "omissions", "notes_with_any_false_diagnosis", "notes_with_no_affirmed_diagnosis")},
        "constrained": {k: complete["keyword"][k] for k in ("micro_precision", "micro_recall", "micro_f1", "omissions", "notes_with_any_false_diagnosis", "notes_with_no_affirmed_diagnosis")},
        "recall_change": complete["paired"]["patient_recall_keyword_minus_no_retrieval"],
    }
    print(json.dumps(brief, indent=2))
if __name__ == "__main__":
    main()
