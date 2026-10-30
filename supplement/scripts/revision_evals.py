import ast
import json
import os
\
import numpy as np
import pandas as pd
\
from claim_scoring import SCP_FOR_CHECK, affirms, filter_note, score_note
from independent_measurements import holds, measurements_for
from rescore_corrected import bootstrap
from retrieval_harm import MEASUREMENT_COLUMNS, facts_for, load_cases, predicate
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
LIBRARY_PATH = os.path.join(ROOT, "knowledge", "ecg_criteria_v2.json")
REPORT = os.path.join(OUT, "independent_factuality.json")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
\
\
def library() -> list[dict]:
    with open(LIBRARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)
def load_jsonl(path: str) -> dict[int, dict]:
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                found[int(item["ecg_id"])] = item
    return found
def scp_table(ecg_ids: set[int]) -> dict[int, set[str]]:
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "scp_codes"])
    found = {}
    for ecg_id, raw in database.loc[database["ecg_id"].isin(ecg_ids), ["ecg_id", "scp_codes"]].itertuples(index=False):
        found[int(ecg_id)] = {code for code, likelihood in ast.literal_eval(raw).items() if likelihood > 0}
    return found
def unig_table() -> pd.DataFrame:
    columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    return pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(columns),
    ).set_index("ecg_id")
def diagnosis_flags(claims: list[dict], facts: dict, labels: dict[str, int]) -> dict:
    low = False
    absent = False
    affirmed = 0
    for claim in claims:
        if claim.get("type") != "diagnosis" or not affirms(claim.get("text", "")):
            continue
        raw = claim.get("label")
        label_list = raw if isinstance(raw, list) else [raw]
        for label in label_list:
            affirmed += 1
            if not isinstance(label, str):
                low = True
                absent = True
                continue
            probability = facts["probabilities"].get(label)
            if probability is None or probability < 0.5:
                low = True
            if labels.get(label, 0) != 1:
                absent = True
    return {"low_probability": low, "absent_from_expert": absent, "n_affirmed": affirmed}
def criterion_flags(claims: list[dict], facts: dict, facts_unig: dict | None, checks: dict[str, str], scp: set[str]) -> dict:
    applied = 0
    no_check = 0
    reject_12 = 0
    support_12 = 0
    reject_unig = 0
    support_unig = 0
    unig_unknown = 0
    linked = 0
    linked_in_scp = 0
    linked_not_in_scp = 0
    for claim in claims:
        if claim.get("type") != "criterion" or not bool(claim.get("applies")):
            continue
        applied += 1
        check = checks.get(claim.get("card_id"))
        if not check:
            no_check += 1
            continue
        if predicate(check, facts):
            support_12 += 1
        else:
            reject_12 += 1
        verdict = None if facts_unig is None else holds(check, facts_unig)
        if verdict is True:
            support_unig += 1
        elif verdict is False:
            reject_unig += 1
        else:
            unig_unknown += 1
        if check in SCP_FOR_CHECK:
            linked += 1
            if SCP_FOR_CHECK[check] & scp:
                linked_in_scp += 1
            else:
                linked_not_in_scp += 1
    return {
        "applied": applied,
        "no_check": no_check,
        "support_12": support_12,
        "reject_12": reject_12,
        "support_unig": support_unig,
        "reject_unig": reject_unig,
        "unig_unknown": unig_unknown,
        "linked": linked,
        "linked_in_scp": linked_in_scp,
        "linked_not_in_scp": linked_not_in_scp,
    }
def rate(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))
def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))
def arm_summary(frame: pd.DataFrame, prefix: str) -> dict:
    return {
        "n": int(len(frame)),
        "diagnosis_below_0.5": rate(frame[f"{prefix}_dx_low"].tolist()),
        "diagnosis_absent_from_expert_label": rate(frame[f"{prefix}_dx_absent"].tolist()),
        "criterion_rejected_by_12sl_or_unchecked": rate(frame[f"{prefix}_crit_12"].tolist()),
        "criterion_rejected_by_unig": rate(frame[f"{prefix}_crit_unig"].tolist()),
        "statement_mismatch": rate(frame[f"{prefix}_statement"].tolist()),
        "mean_affirmed_diagnoses": mean(frame[f"{prefix}_n_affirmed"].tolist()),
        "mean_applied_criteria": mean(frame[f"{prefix}_applied"].tolist()),
        "mean_applied_supported_by_12sl": mean(frame[f"{prefix}_support_12"].tolist()),
        "mean_applied_rejected_by_12sl": mean(frame[f"{prefix}_reject_12"].tolist()),
        "mean_applied_without_check": mean(frame[f"{prefix}_no_check"].tolist()),
        "mean_applied_supported_by_unig": mean(frame[f"{prefix}_support_unig"].tolist()),
        "mean_applied_rejected_by_unig": mean(frame[f"{prefix}_reject_unig"].tolist()),
        "linked_criterion_precision_vs_scp": (
            None
            if frame[f"{prefix}_linked"].sum() == 0
            else float(frame[f"{prefix}_linked_in_scp"].sum() / frame[f"{prefix}_linked"].sum())
        ),
        "linked_criteria_applied": int(frame[f"{prefix}_linked"].sum()),
    }
def paired(frame: pd.DataFrame, left: str, right: str) -> dict:
    return bootstrap(frame[left].to_numpy(), frame[right].to_numpy())
def evaluate(notes: dict[int, dict], cases: pd.DataFrame, cards: list[dict], policy_of) -> dict:
    checks = {card["id"]: card["check"] for card in cards if card.get("check")}
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    unig = unig_table()
    scp_of = scp_table(set(notes))
    rows = []
    withdrawals = {
        "n": 0,
        "no_numeric_check": 0,
        "predicate_false_on_12sl": 0,
        "unig_rejects": 0,
        "unig_missing": 0,
        "unig_supports": 0,
    }
    for ecg_id, record in notes.items():
        source = by_id[ecg_id]
        facts = facts_for(source)
        facts_unig = measurements_for(unig.loc[ecg_id], facts["sex"]) if ecg_id in unig.index else None
        if facts_unig is not None:
            facts_unig["probabilities"] = facts["probabilities"]
        labels = {name: int(source[f"y_{name}"]) for name in CLASSES}
        scp = scp_of.get(ecg_id, set())
        policies = policy_of(record, facts, facts_unig, checks)
        item = {
            "ecg_id": ecg_id,
            "human_validated": int(bool(source["validated_by_human"])),
        }
        for name, claims in policies.items():
            diagnosis = diagnosis_flags(claims, facts, labels)
            criterion = criterion_flags(claims, facts, facts_unig, checks, scp)
            scored = score_note(claims, facts, labels, scp, checks)
            item[f"{name}_dx_low"] = int(diagnosis["low_probability"])
            item[f"{name}_dx_absent"] = int(diagnosis["absent_from_expert"])
            item[f"{name}_n_affirmed"] = diagnosis["n_affirmed"]
            item[f"{name}_crit_12"] = int(criterion["reject_12"] + criterion["no_check"] > 0)
            item[f"{name}_crit_unig"] = int(criterion["reject_unig"] > 0)
            item[f"{name}_statement"] = int(len(scored["statement_errors"]) > 0)
            for key, value in criterion.items():
                item[f"{name}_{key}"] = value
        keyword = policies["keyword"]
        checked = policies["check_12sl"]
        checked_on = {
            claim.get("card_id")
            for claim in checked
            if claim.get("type") == "criterion" and bool(claim.get("applies"))
        }
        for claim in keyword:
            if claim.get("type") != "criterion" or not bool(claim.get("applies")):
                continue
            if claim.get("card_id") in checked_on:
                continue
            withdrawals["n"] += 1
            check = checks.get(claim.get("card_id"))
            if not check:
                withdrawals["no_numeric_check"] += 1
                verdict = None
            else:
                withdrawals["predicate_false_on_12sl"] += 1
                verdict = None if facts_unig is None else holds(check, facts_unig)
            if verdict is True:
                withdrawals["unig_supports"] += 1
            elif verdict is False:
                withdrawals["unig_rejects"] += 1
            else:
                withdrawals["unig_missing"] += 1
        rows.append(item)
    frame = pd.DataFrame(rows)
    names = list(policy_of(next(iter(notes.values())), facts_for(by_id[next(iter(notes))]), None, checks))
    validated = frame[frame["human_validated"] == 1]
    summary = {name: arm_summary(frame, name) for name in names}
    validated_summary = {name: arm_summary(validated, name) for name in names}
    contrasts = {}
    if "keyword" in names and "check_12sl" in names:
        contrasts["release_keyword_minus_no_retrieval"] = paired(frame, "keyword_crit_12", "no_retrieval_crit_12") if "no_retrieval" in names else None
        contrasts["unig_rejection_keyword_minus_12sl_check"] = paired(frame, "keyword_crit_unig", "check_12sl_crit_unig")
        contrasts["criterion_12sl_keyword_minus_both"] = paired(frame, "keyword_crit_12", "check_both_crit_12")
        contrasts["expert_diagnosis_keyword_minus_12sl_check"] = paired(frame, "keyword_dx_absent", "check_12sl_dx_absent")
        contrasts["expert_diagnosis_keyword_minus_no_retrieval"] = paired(
            frame, "keyword_dx_absent", "no_retrieval_dx_absent"
        ) if "no_retrieval" in names else None
    return {
        "scored_notes": int(len(frame)),
        "human_validated_notes": int(len(validated)),
        "arms": summary,
        "human_validated_arms": validated_summary,
        "contrasts": contrasts,
        "withdrawals_by_12sl_check": withdrawals,
    }
def main_policies(record, facts, facts_unig, checks):
    return {
        "no_retrieval": record["no_rag"],
        "keyword": record["with_keyword"],
        "tfidf": record["with_tfidf"],
        "check_12sl": filter_note(record["with_keyword"], facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(record["with_keyword"], facts, facts_unig, checks, "agree"),
    }
def main() -> None:
    cards = library()
    cases = load_cases()
    notes = load_jsonl(os.path.join(OUT, "heldout_generations.jsonl"))
    report = {
        "question": (
            "Does the measurement edit still reduce unsupported criteria when the judge is Uni-G, "
            "and do expert superclass labels move when the edit cannot rewrite diagnoses?"
        ),
        "judge": "Uni-G did not enter the 12SL edit. Expert labels are the PTB-XL superclass indicators, not the 12SL predicate.",
        **evaluate(notes, cases, cards, main_policies),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
