import hashlib
import json
import os

import numpy as np
import pandas as pd

from independent_measurements import holds
from retrieval_harm import MEASUREMENT_COLUMNS, Q_COLUMNS, retrieve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ecg_guard")
LIBRARY = os.path.join(ROOT, "knowledge", "ecg_criteria_v2.json")
FEATURES = os.path.join(ROOT, "ptb-xl-plus", "features")
CASE_RECORD = os.path.join(ROOT, "results", "ptbxl_baseline", "case_record.csv")
FROZEN_RESULTS = os.path.join(ROOT, "results")

# Prespecified before any fold-10 look: abstain when the 12SL value is within
# the fold-9 95th percentile of |12SL - Uni-G| of the published threshold.
MARGIN_PERCENTILE = 95

SCALAR = {
    "qrs_ge_120": ("QRS_Dur_Global", "ge", 120, "qrs_ms"),
    "qrs_lt_110": ("QRS_Dur_Global", "lt", 110, "qrs_ms"),
    "pr_ge_200": ("PR_Int_Global", "ge", 200, "pr_ms"),
    "pr_lt_120": ("PR_Int_Global", "lt", 120, "pr_ms"),
    "axis_le_m30": ("R_AxisFrontal_Global", "le", -30, "axis_deg"),
    "axis_ge_90": ("R_AxisFrontal_Global", "ge", 90, "axis_deg"),
    "hr_ge_100": ("HR_Ventr_Global", "ge", 100, "hr_bpm"),
    "hr_lt_60": ("HR_Ventr_Global", "lt", 60, "hr_bpm"),
    "st_ii_le_m0_1": ("ST_Amp_II", "le", -0.1, "st_mv"),
    "st_ii_ge_01": ("ST_Amp_II", "ge", 0.1, "st_mv"),
    "qtc_long": ("QT_IntBazett_Global", "ge", None, "qtc_ms"),
    "sokolow_ge_3_5": ("sokolow", "ge", 3.5, "sokolow_mv"),
    "sokolow_lt_35": ("sokolow", "lt", 3.5, "sokolow_mv"),
    "q_ge_40": ("q_any", "ge", 40, "q_ms"),
}


def library() -> list[dict]:
    with open(LIBRARY, encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest() -> dict:
    files = []
    for dirpath, _, names in os.walk(FROZEN_RESULTS):
        if os.path.abspath(dirpath).startswith(os.path.abspath(OUT)):
            continue
        for name in sorted(names):
            if not name.endswith((".json", ".jsonl", ".csv")):
                continue
            path = os.path.join(dirpath, name)
            files.append({
                "path": os.path.relpath(path, ROOT).replace("\\", "/"),
                "bytes": os.path.getsize(path),
                "sha256": sha256(path),
            })
    manifest = {
        "status": "frozen",
        "meaning": "Existing PTB-XL notes, scores, EEG, and sleep outputs are not inputs to a new threshold and are not rewritten by ECG-GUARD.",
        "files": files,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "freeze_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return {"n_files": len(files)}


def sex_label(value) -> str:
    return "male" if int(value) == 0 else "female"


def sokolow(values: dict):
    parts = [values.get(name) for name in ("S_Amp_V1", "R_Amp_V5", "R_Amp_V6")]
    if any(part is None or (isinstance(part, float) and np.isnan(part)) for part in parts):
        return None
    return abs(float(parts[0])) + max(float(parts[1]), float(parts[2]))


def q_values(values: dict) -> list[float]:
    present = []
    for name in Q_COLUMNS:
        value = values.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        present.append(float(value))
    return present


def deciding_value(check: str, facts: dict):
    values = facts["measurements"]
    kind = SCALAR[check][0]
    if kind == "sokolow":
        return sokolow(values)
    if kind == "q_any":
        present = q_values(values)
        if len(present) != len(Q_COLUMNS):
            return None
        return max(present) if present else None
    value = values.get(kind)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def threshold_for(check: str, facts: dict) -> float:
    limit = SCALAR[check][2]
    if check == "qtc_long":
        return 450.0 if facts["sex"] == "male" else 460.0
    return float(limit)


def scalar_status(check: str, facts: dict, margin: float) -> str:
    if check == "q_ge_40":
        present = q_values(facts["measurements"])
        if not present:
            return "indeterminate"
        if len(present) != len(Q_COLUMNS) and max(present) <= 40 + margin:
            return "indeterminate"
        value = max(present)
    else:
        value = deciding_value(check, facts)
        if value is None:
            return "indeterminate"
    limit = threshold_for(check, facts)
    direction = SCALAR[check][1]
    if abs(value - limit) <= margin:
        return "indeterminate"
    above = value > limit
    if direction in ("ge",):
        return "supported" if above else "refuted"
    if direction == "le":
        return "supported" if value < limit else "refuted"
    if direction == "lt":
        return "supported" if value < limit else "refuted"
    raise KeyError(check)


def required_present(check: str, facts: dict) -> bool:
    if check is None:
        return False
    values = facts["measurements"]
    kind = SCALAR[check][0]
    if kind == "sokolow":
        return sokolow(values) is not None
    if kind == "q_any":
        return len(q_values(values)) == len(Q_COLUMNS)
    value = values.get(kind)
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def verify(check: str, facts_12: dict, facts_unig: dict | None, margins: dict) -> str:
    if not check or not required_present(check, facts_12):
        return "indeterminate"
    margin = margins[SCALAR[check][3]]
    status = scalar_status(check, facts_12, margin)
    if status == "indeterminate" or facts_unig is None:
        return status
    left = holds(check, facts_12)
    right = holds(check, facts_unig)
    if left is None or right is None or left != right:
        return "indeterminate"
    return status


def facts_from(row: pd.Series, measurements: dict) -> dict:
    asserted = [name for name in ("NORM", "MI", "STTC", "CD", "HYP") if row[f"p_{name}"] >= 0.5]
    return {
        "sex": sex_label(row["sex"]),
        "asserted_labels": asserted,
        "measurements": measurements,
    }


def load_features(name: str, ids: set[int]) -> pd.DataFrame:
    path = os.path.join(FEATURES, f"{name}_features.csv")
    header = pd.read_csv(path, nrows=0).columns
    wanted = ["ecg_id", *MEASUREMENT_COLUMNS, "QRS_AxisFront_Global"]
    use = [column for column in wanted if column in header]
    frame = pd.read_csv(path, usecols=use)
    frame = frame[frame["ecg_id"].isin(ids)].set_index("ecg_id")
    return frame


def measurement_dict(row: pd.Series) -> dict:
    values = {}
    for column in row.index:
        value = row[column]
        if pd.notna(value):
            values[column] = float(value)
    if "QRS_AxisFront_Global" in values and "R_AxisFrontal_Global" not in values:
        values["R_AxisFrontal_Global"] = values["QRS_AxisFront_Global"]
    return values


def margins_from(pairs: dict[str, list[float]]) -> dict:
    margins = {}
    for name, values in pairs.items():
        if not values:
            raise RuntimeError(f"No calibration pairs for {name}")
        margins[name] = float(np.percentile(np.abs(values), MARGIN_PERCENTILE))
    return margins


def collect_pairs(sl: pd.DataFrame, unig: pd.DataFrame) -> dict[str, list[float]]:
    pairs = {name: [] for name in ("qrs_ms", "pr_ms", "axis_deg", "hr_bpm", "st_mv", "qtc_ms", "sokolow_mv", "q_ms")}
    shared = sl.index.intersection(unig.index)
    for ecg_id in shared:
        left = measurement_dict(sl.loc[ecg_id])
        right = measurement_dict(unig.loc[ecg_id])
        for key, column in (
            ("qrs_ms", "QRS_Dur_Global"),
            ("pr_ms", "PR_Int_Global"),
            ("hr_bpm", "HR_Ventr_Global"),
            ("st_mv", "ST_Amp_II"),
            ("qtc_ms", "QT_IntBazett_Global"),
        ):
            if column in left and column in right:
                pairs[key].append(left[column] - right[column])
        if "R_AxisFrontal_Global" in left and "R_AxisFrontal_Global" in right:
            pairs["axis_deg"].append(left["R_AxisFrontal_Global"] - right["R_AxisFrontal_Global"])
        left_voltage = sokolow(left)
        right_voltage = sokolow(right)
        if left_voltage is not None and right_voltage is not None:
            pairs["sokolow_mv"].append(left_voltage - right_voltage)
        left_q = q_values(left)
        right_q = q_values(right)
        if len(left_q) == len(Q_COLUMNS) and len(right_q) == len(Q_COLUMNS):
            pairs["q_ms"].append(max(left_q) - max(right_q))
    return pairs


def counterfactual(margins: dict) -> dict:
    margin = margins["qrs_ms"]
    unrelated = "hr_lt_60"
    cases = []
    for qrs in (80, 110, 118, 120, 132, 160, None):
        measurements = {"QRS_Dur_Global": qrs, "HR_Ventr_Global": 70}
        measurements = {key: value for key, value in measurements.items() if value is not None}
        facts = {"sex": "male", "asserted_labels": ["CD"], "measurements": measurements}
        qrs_status = verify("qrs_ge_120", facts, facts, margins)
        rate_status = verify(unrelated, facts, facts, margins)
        if qrs is None:
            expected = "indeterminate"
        elif abs(qrs - 120) <= margin:
            expected = "indeterminate"
        elif qrs > 120:
            expected = "supported"
        else:
            expected = "refuted"
        if qrs_status != expected or rate_status != "refuted":
            raise RuntimeError(f"Counterfactual failed for QRS {qrs}: {qrs_status}, {rate_status}")
        cases.append({"qrs_ms": qrs, "wide_qrs": qrs_status, "unrelated_bradycardia": rate_status, "expected": expected})
    return {"qrs_margin_ms": margin, "cases": cases, "passed": True}


def main() -> None:
    frozen = freeze_manifest()
    cards = library()
    numeric = [card for card in cards if card.get("check")]
    record = pd.read_csv(CASE_RECORD)
    fold = record[record["strat_fold"].eq(9)].sort_values("ecg_id").drop_duplicates("patient_id")
    ids = set(fold["ecg_id"].astype(int))
    sl = load_features("12sl", ids)
    unig = load_features("unig", ids)
    margins = margins_from(collect_pairs(sl, unig))
    synthetic = counterfactual(margins)

    keyword_untestable = []
    numeric_missing = []
    guard_untestable = []
    old_unig_false = []
    guard_unig_false = []
    old_missing_as_false = []
    guard_indeterminate = []
    agreed_far = []
    agreed_far_kept_old = []
    agreed_far_kept_guard = []
    agreed_near_abstained = []
    n_keyword = []
    n_guard = []
    deterministic_supported = []

    for row in fold.itertuples(index=False):
        ecg_id = int(row.ecg_id)
        if ecg_id not in sl.index or ecg_id not in unig.index:
            continue
        series = pd.Series(row._asdict())
        facts_12 = facts_from(series, measurement_dict(sl.loc[ecg_id]))
        facts_unig = facts_from(series, measurement_dict(unig.loc[ecg_id]))
        keyword = retrieve(facts_12, cards, k=3)
        numeric_cards = retrieve(facts_12, numeric, k=3)
        testable = [card for card in numeric if required_present(card["check"], facts_12)]
        guarded = retrieve(facts_12, testable, k=3) if testable else []
        n_keyword.append(len(keyword))
        n_guard.append(len(guarded))
        keyword_untestable.append(int(any(not card.get("check") or not required_present(card["check"], facts_12) for card in keyword)))
        numeric_missing.append(int(any(not required_present(card["check"], facts_12) for card in numeric_cards)))
        guard_untestable.append(int(any(not required_present(card["check"], facts_12) for card in guarded)))

        old_bad = False
        guard_bad = False
        missing_withdrawn = False
        saw_indeterminate = False
        for card in numeric_cards:
            check = card["check"]
            left = holds(check, facts_12)
            right = holds(check, facts_unig)
            if left is None:
                missing_withdrawn = True
            elif left is False:
                pass
            elif right is False:
                old_bad = True
            status = verify(check, facts_12, facts_unig, margins)
            if status == "supported" and right is False:
                guard_bad = True
            if status == "indeterminate":
                saw_indeterminate = True
            far = left is True and right is True and status == "supported"
            near = left is True and right is True and status == "indeterminate"
            if left is True and right is True and scalar_status(check, facts_12, margins[SCALAR[check][3]]) != "indeterminate":
                agreed_far.append(1)
                agreed_far_kept_old.append(1)
                agreed_far_kept_guard.append(int(status == "supported"))
            elif near:
                agreed_near_abstained.append(1)
            elif far:
                pass
        old_unig_false.append(int(old_bad))
        guard_unig_false.append(int(guard_bad))
        old_missing_as_false.append(int(missing_withdrawn))
        guard_indeterminate.append(int(saw_indeterminate))
        deterministic_supported.append(sum(
            verify(card["check"], facts_12, facts_unig, margins) == "supported" for card in numeric
        ))

    def rate(values: list) -> float:
        return float(np.mean(values)) if values else None

    report = {
        "name": "ECG-GUARD development check",
        "fold": 9,
        "unit": "one electrocardiogram per patient",
        "n_patients": len(old_unig_false),
        "margins_95th_abs_12sl_minus_unig": margins,
        "retrieval": {
            "keyword_top3_contains_untestable_card": rate(keyword_untestable),
            "numeric_only_top3_contains_missing_measurement": rate(numeric_missing),
            "guard_top3_contains_untestable_card": rate(guard_untestable),
            "mean_keyword_cards": rate(n_keyword),
            "mean_guard_cards": rate(n_guard),
        },
        "if_every_retrieved_numeric_card_is_applied": {
            "old_filter_patient_rate_unig_rejects_a_kept_card": rate(old_unig_false),
            "guard_patient_rate_unig_rejects_a_supported_card": rate(guard_unig_false),
            "old_filter_patient_rate_missing_measurement_withdrawn_as_not_established": rate(old_missing_as_false),
            "guard_patient_rate_any_indeterminate": rate(guard_indeterminate),
            "clear_agreements_kept_by_old_filter": rate(agreed_far_kept_old),
            "clear_agreements_kept_by_guard": rate(agreed_far_kept_guard),
            "n_clear_agreements": len(agreed_far),
        },
        "deterministic_measurement_to_text": {
            "mean_supported_criteria": rate(deterministic_supported),
            "note": "Supported requires both algorithms and a value outside the calibration margin. A false assertion against 12SL is zero by construction.",
        },
        "counterfactual_qrs": synthetic,
        "not_run": [
            "LUDB cardiologist waveform reference",
            "MIMIC-IV-ECG external sample",
            "PhysioNet Challenge 2021",
            "Qwen3-14B, Gemma 3 12B, MedGemma",
            "New language-model notes",
        ],
        "frozen_files": frozen,
    }
    with open(os.path.join(OUT, "development_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in ("n_patients", "retrieval", "if_every_retrieved_numeric_card_is_applied", "counterfactual_qrs")}, indent=2))


if __name__ == "__main__":
    main()
