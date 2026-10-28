import json
import os

from diagnosis_prf import gold_of, predicted
from exploratory_revision import as_claims
from retrieval_harm import facts_for, load_cases
from revision_evals import load_jsonl

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
LOST = [4458, 6180, 6625, 7670, 7916, 8524, 8880, 9024, 10388, 10837, 17690]


def slim(claims):
    rows = []
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        rows.append({
            "type": claim.get("type"),
            "label": claim.get("label"),
            "card_id": claim.get("card_id"),
            "applies": claim.get("applies"),
            "text": claim.get("text"),
            "counted": sorted(predicted([claim])),
        })
    return rows


def recall(notes, cases):
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    no = key = n = 0
    for ecg_id, record in notes.items():
        row = by_id.get(int(ecg_id))
        if row is None or int(row["y_HYP"]) != 1:
            continue
        n += 1
        no += int("HYP" in predicted(as_claims(record.get("no_rag"))))
        key += int("HYP" in predicted(as_claims(record.get("with_keyword"))))
    return {"n": n, "no_cards": no, "keyword": key, "recall_no": no / n, "recall_key": key / n}


def main():
    cases = load_cases()
    matched = load_jsonl(os.path.join(OUT, "matched_constrained_generations.jsonl"))
    fair = load_jsonl(os.path.join(OUT, "fair_generations.jsonl"))
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    reads = []
    for ecg_id in LOST:
        record = matched[ecg_id]
        row = by_id[ecg_id]
        reads.append({
            "ecg_id": ecg_id,
            "p_hyp": round(float(row["p_HYP"]), 3),
            "no_rag": slim(as_claims(record.get("no_rag"))),
            "keyword": slim(as_claims(record.get("with_keyword"))),
        })
    report = {
        "matched_schema_differs": recall(matched, cases),
        "fair_identical_json_object": recall(fair, cases),
        "fair_n": len(fair),
        "reads": reads,
    }
    path = os.path.join(OUT, "hyp_read.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("matched_schema_differs", "fair_identical_json_object", "fair_n")}, indent=2))


if __name__ == "__main__":
    main()
