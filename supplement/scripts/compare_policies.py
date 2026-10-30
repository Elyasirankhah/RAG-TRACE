import json
import os
\
import pandas as pd
\
from retrieval_harm import cards, facts_for, load_cases, predicate
\
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "ptbxl_baseline", "rag_generations.jsonl")
OUT = os.path.join(ROOT, "results", "ptbxl_baseline", "policy_comparison.json")
\
\
def load_records() -> list[dict]:
    latest = {}
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                latest[item["ecg_id"]] = item
    return list(latest.values())
def patient_errors(claims: list[dict], facts: dict, checks: dict[str, str]) -> dict:
    unsupported = []
    supported_labels = set()
    applied_true = set()
    for claim in claims:
        kind = claim.get("type")
        if kind == "diagnosis":
            label = claim.get("label")
            probability = facts["probabilities"].get(label)
            if probability is None or probability < 0.5:
                unsupported.append("diagnosis_without_probability")
            else:
                supported_labels.add(label)
        elif kind == "criterion":
            card_id = claim.get("card_id")
            if card_id not in checks:
                unsupported.append("unknown_criterion")
                continue
            holds = predicate(checks[card_id], facts)
            applies = bool(claim.get("applies"))
            if applies and not holds:
                unsupported.append("criterion_not_supported")
            elif applies and holds:
                applied_true.add(card_id)
    return {
        "unsupported": unsupported,
        "supported_labels": supported_labels,
        "applied_true": applied_true,
    }
def guarded_claims(record: dict, library: dict[str, dict], facts: dict) -> list[dict]:
    claims = [claim for claim in record["no_rag"] if claim.get("type") != "criterion"]
    for item in record["retrieved"]:
        card = library[item["id"]]
        holds = predicate(card["check"], facts)
        if holds:
            text = f"This patient's measurements meet the criterion: {card['text']}"
        else:
            text = f"Retrieved reference, not applied to this patient: {card['text']}"
        claims.append({"type": "criterion", "card_id": card["id"], "applies": holds, "text": text})
    return claims
def summarize(rows: list[dict]) -> dict:
    def rate(name: str) -> float:
        return float(sum(row[name] for row in rows) / len(rows))
    return {
        "n": len(rows),
        "any_unsupported_rate": {name: rate(f"{name}_any") for name in ("no_retrieval", "always_retrieve", "measurement_check")},
        "mean_unsupported_claims": {name: rate(f"{name}_count") for name in ("no_retrieval", "always_retrieve", "measurement_check")},
        "supported_diagnosis_retention": {name: rate(f"{name}_retention") for name in ("no_retrieval", "always_retrieve", "measurement_check")},
        "true_criteria_applied_rate": {name: rate(f"{name}_true_applied") for name in ("always_retrieve", "measurement_check")},
        "harm_vs_clean_no_retrieval": {name: rate(f"{name}_harm") for name in ("always_retrieve", "measurement_check")},
    }
def main() -> None:
    library = {card["id"]: card for card in cards()}
    checks = {card_id: card["check"] for card_id, card in library.items()}
    by_id = {int(row["ecg_id"]): row for _, row in load_cases().iterrows()}
    scored = []
    for record in load_records():
        facts = facts_for(by_id[record["ecg_id"]])
        policies = {
            "no_retrieval": record["no_rag"],
            "always_retrieve": record["with_rag"],
            "measurement_check": guarded_claims(record, library, facts),
        }
        expected = {label for label, probability in facts["probabilities"].items() if probability >= 0.5}
        held = {item["id"] for item in record["retrieved"] if predicate(checks[item["id"]], facts)}
        row = {"split": record["split"], "clean_no_retrieval": False}
        for name, claims in policies.items():
            result = patient_errors(claims, facts, checks)
            row[f"{name}_count"] = len(result["unsupported"])
            row[f"{name}_any"] = int(len(result["unsupported"]) > 0)
            row[f"{name}_retention"] = 1.0 if not expected else len(expected & result["supported_labels"]) / len(expected)
            if name != "no_retrieval":
                row[f"{name}_true_applied"] = 1.0 if not held else len(held & result["applied_true"]) / len(held)
        row["clean_no_retrieval"] = row["no_retrieval_count"] == 0
        row["always_retrieve_harm"] = int(row["clean_no_retrieval"] and row["always_retrieve_count"] > 0)
        row["measurement_check_harm"] = int(row["clean_no_retrieval"] and row["measurement_check_count"] > 0)
        scored.append(row)
    frame = pd.DataFrame(scored)
    summary = {
        "rule": "A retrieved criterion is stated as true for this patient only when the 12SL measurements meet it.",
        "test": summarize(frame[frame["split"] == "test"].to_dict("records")),
        "development": summarize(frame[frame["split"] == "calibrate"].to_dict("records")),
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
