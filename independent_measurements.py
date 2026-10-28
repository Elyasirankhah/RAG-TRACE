import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from retrieval_harm import Q_COLUMNS, cards

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "independent_measurements.json")
FEATURES = os.path.join(ROOT, "ptb-xl-plus", "features")

CHECK_OF = {card["id"]: card["check"] for card in cards()}
LEAD_Q = ["II", "III", "aVF", "V1", "V2", "V3", "V4"]


def read_source(name: str, columns: list[str]) -> pd.DataFrame:
    path = os.path.join(FEATURES, f"{name}_features.csv")
    header = pd.read_csv(path, nrows=0).columns
    use = ["ecg_id", *[column for column in columns if column in header]]
    frame = pd.read_csv(path, usecols=use)
    ids = set(pd.read_csv(os.path.join(OUT, "confirm_sample.csv"))["ecg_id"].astype(int))
    return frame[frame["ecg_id"].isin(ids)].set_index("ecg_id")


def measurements_for(row: pd.Series, sex: str) -> dict:
    values = {}
    for column in row.index:
        value = row[column]
        if pd.notna(value):
            values[column] = float(value)
    if "QRS_AxisFront_Global" in values and "R_AxisFrontal_Global" not in values:
        values["R_AxisFrontal_Global"] = values["QRS_AxisFront_Global"]
    return {"sex": sex, "measurements": values}


def holds(check: str, facts: dict):
    """Frozen threshold. None means a required measurement is absent."""
    values = facts["measurements"]

    def read(name: str):
        if name not in values:
            return None
        return values[name]

    if check == "qrs_ge_120":
        value = read("QRS_Dur_Global")
        return None if value is None else value >= 120
    if check == "pr_ge_200":
        value = read("PR_Int_Global")
        return None if value is None else value >= 200
    if check == "axis_le_m30":
        value = read("R_AxisFrontal_Global")
        return None if value is None else value <= -30
    if check == "sokolow_ge_3_5":
        parts = [read(name) for name in ("S_Amp_V1", "R_Amp_V5", "R_Amp_V6")]
        if any(part is None for part in parts):
            return None
        return abs(parts[0]) + max(parts[1], parts[2]) >= 3.5
    if check == "qtc_long":
        value = read("QT_IntBazett_Global")
        if value is None:
            return None
        limit = 450 if facts["sex"] == "male" else 460
        return value >= limit
    if check == "st_ii_le_m0_1":
        value = read("ST_Amp_II")
        return None if value is None else value <= -0.1
    if check == "q_ge_40":
        present = [read(name) for name in Q_COLUMNS]
        if all(value is None for value in present):
            return None
        return any((value or 0) >= 40 for value in present)
    if check == "hr_ge_100":
        value = read("HR_Ventr_Global")
        return None if value is None else value >= 100
    if check == "qrs_lt_110":
        value = read("QRS_Dur_Global")
        return None if value is None else value < 110
    if check == "pr_lt_120":
        value = read("PR_Int_Global")
        return None if value is None else value < 120
    if check == "axis_ge_90":
        value = read("R_AxisFrontal_Global")
        return None if value is None else value >= 90
    if check == "sokolow_lt_35":
        parts = [read(name) for name in ("S_Amp_V1", "R_Amp_V5", "R_Amp_V6")]
        if any(part is None for part in parts):
            return None
        return abs(parts[0]) + max(parts[1], parts[2]) < 3.5
    if check == "hr_lt_60":
        value = read("HR_Ventr_Global")
        return None if value is None else value < 60
    if check == "st_ii_ge_01":
        value = read("ST_Amp_II")
        return None if value is None else value >= 0.1
    raise KeyError(check)


def model_applies(claims: list[dict], card_id: str) -> bool:
    return any(claim.get("type") == "criterion" and claim.get("card_id") == card_id and bool(claim.get("applies")) for claim in claims)


def diagnosis_unsupported(claims: list[dict], probabilities: dict) -> int:
    count = 0
    for claim in claims:
        if claim.get("type") != "diagnosis":
            continue
        probability = probabilities.get(claim.get("label"))
        if probability is None or probability < 0.5:
            count += 1
    return count


def kappa(left: np.ndarray, right: np.ndarray) -> float | None:
    known = ~pd.isna(left) & ~pd.isna(right)
    if known.sum() == 0:
        return None
    a = left[known].astype(bool)
    b = right[known].astype(bool)
    po = float(np.mean(a == b))
    p_yes = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if p_yes == 1:
        return 1.0
    return float((po - p_yes) / (1 - p_yes))


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(2)
    draws = []
    n = len(a)
    for _ in range(10_000):
        index = rng.integers(0, n, n)
        draws.append(float(a[index].mean() - b[index].mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    discordant = only_a + only_b
    return {
        "difference": float(a.mean() - b.mean()),
        "ci95": [float(low), float(high)],
        "n_a_only": only_a,
        "n_b_only": only_b,
        "exact_mcnemar_p": float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else None,
    }


def numeric_agreement(left: pd.Series, right: pd.Series) -> dict:
    both = pd.concat([left, right], axis=1, keys=["left", "right"]).dropna()
    if both.empty:
        return {"n": 0}
    diff = both["left"] - both["right"]
    return {
        "n": int(len(both)),
        "median_abs_diff": float(diff.abs().median()),
        "mean_left": float(both["left"].mean()),
        "mean_right": float(both["right"].mean()),
        "pearson_r": float(both["left"].corr(both["right"])),
    }


def main() -> None:
    sample = pd.read_csv(os.path.join(OUT, "confirm_sample.csv"))
    meta = pd.read_csv(
        os.path.join(OUT, "case_record.csv"),
        usecols=["ecg_id", "sex", "signal_quality_issue", "p_NORM", "p_MI", "p_STTC", "p_CD", "p_HYP"],
    )
    meta = meta[meta["ecg_id"].isin(sample["ecg_id"])].set_index("ecg_id")
    needed = [
        "QRS_Dur_Global",
        "PR_Int_Global",
        "R_AxisFrontal_Global",
        "QRS_AxisFront_Global",
        "S_Amp_V1",
        "R_Amp_V5",
        "R_Amp_V6",
        "QT_IntBazett_Global",
        "ST_Amp_II",
        *Q_COLUMNS,
        "HR_Ventr_Global",
        "QT_Int_Global",
        "QT_IntFramingham_Global",
        "RR_Mean_Global",
        "QRS_Dur_II",
        "PR_Int_II",
    ]
    sl = read_source("12sl", needed)
    unig = read_source("unig", needed)
    deli = read_source("ecgdeli", needed)

    sexes = {int(ecg_id): ("male" if int(meta.loc[ecg_id, "sex"]) == 0 else "female") for ecg_id in sample["ecg_id"]}
    facts = {
        "12sl": {ecg_id: measurements_for(sl.loc[ecg_id], sexes[ecg_id]) for ecg_id in sl.index},
        "unig": {ecg_id: measurements_for(unig.loc[ecg_id], sexes[ecg_id]) for ecg_id in unig.index},
        "ecgdeli": {ecg_id: measurements_for(deli.loc[ecg_id], sexes[ecg_id]) for ecg_id in deli.index},
    }
    checks = list(dict.fromkeys(CHECK_OF.values()))
    agreement = {}
    for check in checks:
        left, right = [], []
        for ecg_id in sample["ecg_id"].astype(int):
            left.append(holds(check, facts["12sl"][ecg_id]))
            right.append(holds(check, facts["unig"][ecg_id]))
        left_arr = np.array(left, dtype=object)
        right_arr = np.array(right, dtype=object)
        known = np.array([item is not None for item in left]) & np.array([item is not None for item in right])
        a = np.array([bool(item) for item in left_arr[known]])
        b = np.array([bool(item) for item in right_arr[known]])
        agreement[check] = {
            "n_both_known": int(known.sum()),
            "n_missing": int((~known).sum()),
            "n_agree": int(np.sum(a == b)),
            "n_12sl_only": int(np.sum(a & ~b)),
            "n_unig_only": int(np.sum(~a & b)),
            "agreement": float(np.mean(a == b)) if known.any() else None,
            "kappa": kappa(pd.Series(a), pd.Series(b)) if known.any() else None,
        }

    records = {}
    with open(os.path.join(OUT, "confirm_generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[int(item["ecg_id"])] = item

    rows = []
    for ecg_id, record in records.items():
        probabilities = {label: float(meta.loc[ecg_id, f"p_{label}"]) for label in ["NORM", "MI", "STTC", "CD", "HYP"]}
        diagnoses = diagnosis_unsupported(record["with_rag"], probabilities)
        false_counts = {"always_retrieve_vs_12sl": 0, "always_retrieve_vs_unig": 0, "check_12sl_vs_unig": 0, "check_unig_vs_12sl": 0}
        missed = {"check_12sl_vs_unig": 0, "check_unig_vs_12sl": 0}
        evaluable = {"12sl": 0, "unig": 0}
        for item in record["retrieved"]:
            check = CHECK_OF[item["id"]]
            stated = model_applies(record["with_rag"], item["id"])
            by_source = {source: holds(check, facts[source][ecg_id]) for source in ("12sl", "unig")}
            if by_source["12sl"] is not None:
                evaluable["12sl"] += 1
                false_counts["always_retrieve_vs_12sl"] += int(stated and not by_source["12sl"])
                false_counts["check_unig_vs_12sl"] += int(by_source["unig"] is True and by_source["12sl"] is False)
                missed["check_unig_vs_12sl"] += int(by_source["12sl"] is True and by_source["unig"] is not True)
            if by_source["unig"] is not None:
                evaluable["unig"] += 1
                false_counts["always_retrieve_vs_unig"] += int(stated and not by_source["unig"])
                false_counts["check_12sl_vs_unig"] += int(by_source["12sl"] is True and by_source["unig"] is False)
                missed["check_12sl_vs_unig"] += int(by_source["unig"] is True and by_source["12sl"] is not True)
        row = {
            "ecg_id": ecg_id,
            "signal_quality_issue": int(meta.loc[ecg_id, "signal_quality_issue"]),
            "diagnosis_unsupported": diagnoses,
            "evaluable_12sl": evaluable["12sl"],
            "evaluable_unig": evaluable["unig"],
        }
        for name, count in {**false_counts, **{f"miss_{key}": value for key, value in missed.items()}}.items():
            row[name] = count
            row[f"{name}_any"] = int(count > 0)
        row["primary_always_vs_unig"] = int(diagnoses + false_counts["always_retrieve_vs_unig"] > 0)
        row["primary_check12_vs_unig"] = int(diagnoses + false_counts["check_12sl_vs_unig"] > 0)
        row["primary_check_unig_vs_12sl"] = int(diagnoses + false_counts["check_unig_vs_12sl"] > 0)
        row["primary_always_vs_12sl"] = int(diagnoses + false_counts["always_retrieve_vs_12sl"] > 0)
        rows.append(row)
    frame = pd.DataFrame(rows)

    def rates(part: pd.DataFrame) -> dict:
        return {
            "n": int(len(part)),
            "criterion_withheld_when_reference_met_case_rate": {
                "check_12sl_reference_unig": float(part["miss_check_12sl_vs_unig_any"].mean()),
                "check_unig_reference_12sl": float(part["miss_check_unig_vs_12sl_any"].mean()),
            },
            "criterion_false_application_case_rate": {
                "always_retrieve_reference_12sl": float(part["always_retrieve_vs_12sl_any"].mean()),
                "always_retrieve_reference_unig": float(part["always_retrieve_vs_unig_any"].mean()),
                "check_12sl_reference_unig": float(part["check_12sl_vs_unig_any"].mean()),
                "check_unig_reference_12sl": float(part["check_unig_vs_12sl_any"].mean()),
            },
            "primary_case_rate_diagnosis_plus_criterion": {
                "always_retrieve_reference_12sl": float(part["primary_always_vs_12sl"].mean()),
                "always_retrieve_reference_unig": float(part["primary_always_vs_unig"].mean()),
                "check_12sl_reference_unig": float(part["primary_check12_vs_unig"].mean()),
                "check_unig_reference_12sl": float(part["primary_check_unig_vs_12sl"].mean()),
            },
        }

    hr_from_rr = 60_000 / deli["RR_Mean_Global"]
    report = {
        "confirmatory_ecgs": int(len(sample)),
        "scored_notes": int(len(frame)),
        "rule": "Uni-G supplies the same eight criteria. ECGDeli does not supply equivalent values, so its criteria are withheld.",
        "numeric_agreement_with_12sl": {
            "unig_qrs_ms": numeric_agreement(sl["QRS_Dur_Global"], unig["QRS_Dur_Global"]),
            "unig_pr_ms": numeric_agreement(sl["PR_Int_Global"], unig["PR_Int_Global"]),
            "unig_qtc_bazett_ms": numeric_agreement(sl["QT_IntBazett_Global"], unig["QT_IntBazett_Global"]),
            "unig_hr_bpm": numeric_agreement(sl["HR_Ventr_Global"], unig["HR_Ventr_Global"]),
            "unig_st_ii_mv": numeric_agreement(sl["ST_Amp_II"], unig["ST_Amp_II"]),
            "unig_s_v1_mv": numeric_agreement(sl["S_Amp_V1"], unig["S_Amp_V1"]),
            "unig_axis_degrees": numeric_agreement(sl["R_AxisFrontal_Global"], unig["QRS_AxisFront_Global"]),
            "ecgdeli_qrs_ms": numeric_agreement(sl["QRS_Dur_Global"], deli["QRS_Dur_Global"]),
            "ecgdeli_pr_ms": numeric_agreement(sl["PR_Int_Global"], deli["PR_Int_Global"]),
            "ecgdeli_s_v1_mv": numeric_agreement(sl["S_Amp_V1"], deli["S_Amp_V1"]),
            "ecgdeli_r_v5_mv": numeric_agreement(sl["R_Amp_V5"], deli["R_Amp_V5"]),
            "ecgdeli_hr_from_rr_bpm": numeric_agreement(sl["HR_Ventr_Global"], hr_from_rr),
        },
        "criterion_agreement_12sl_vs_unig": agreement,
        "scored_notes_summary": rates(frame),
        "signal_quality_issue": rates(frame[frame["signal_quality_issue"] == 1]),
        "no_signal_quality_issue": rates(frame[frame["signal_quality_issue"] == 0]),
        "paired_tests": {
            "reference_unig_always_minus_12sl_check": paired_difference(
                frame["primary_always_vs_unig"].to_numpy(), frame["primary_check12_vs_unig"].to_numpy()
            ),
            "reference_12sl_always_minus_unig_check": paired_difference(
                frame["primary_always_vs_12sl"].to_numpy(), frame["primary_check_unig_vs_12sl"].to_numpy()
            ),
            "criterion_only_reference_unig_always_minus_12sl_check": paired_difference(
                frame["always_retrieve_vs_unig_any"].to_numpy(), frame["check_12sl_vs_unig_any"].to_numpy()
            ),
            "criterion_only_reference_12sl_always_minus_unig_check": paired_difference(
                frame["always_retrieve_vs_12sl_any"].to_numpy(), frame["check_unig_vs_12sl_any"].to_numpy()
            ),
        },
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
