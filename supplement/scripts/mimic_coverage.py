import json
import os
\
import numpy as np
import pandas as pd
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "mimic_coverage.json")
MIMIC = os.path.join(ROOT, "mimic-iv-ecg", "machine_measurements.csv")
\
\
def plausible(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.where(series.between(low, high))
def rate(mask: pd.Series) -> dict:
    known = mask.notna()
    return {
        "n_known": int(known.sum()),
        "n_missing": int((~known).sum()),
        "positive_rate": float(mask[known].mean()) if known.any() else None,
    }
def main() -> None:
    mimic = pd.read_csv(
        MIMIC,
        usecols=["subject_id", "study_id", "rr_interval", "p_onset", "qrs_onset", "qrs_end", "qrs_axis"],
    )
    mimic["qrs_ms"] = plausible(mimic["qrs_end"] - mimic["qrs_onset"], 40, 250)
    mimic["pr_ms"] = plausible(mimic["qrs_onset"] - mimic["p_onset"], 80, 400)
    mimic["hr"] = plausible(60_000 / mimic["rr_interval"], 20, 300)
    mimic["axis"] = plausible(mimic["qrs_axis"], -180, 180)
    rng = np.random.default_rng(5)
    one = (
        mimic.sample(frac=1, random_state=int(rng.integers(0, 10_000)))
        .drop_duplicates("subject_id")
    )
    sl = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "12sl_features.csv"),
        usecols=["ecg_id", "QRS_Dur_Global", "PR_Int_Global", "HR_Ventr_Global", "R_AxisFrontal_Global"],
    )
    cases = pd.read_csv(os.path.join(OUT, "case_record.csv"), usecols=["ecg_id", "split", "patient_id"])
    test_ids = set(cases.loc[cases["split"] == "test", "ecg_id"])
    ptb = sl[sl["ecg_id"].isin(test_ids)].copy()
    ptb = ptb.merge(cases[["ecg_id", "patient_id"]], on="ecg_id")
    ptb = ptb.sample(frac=1, random_state=5).drop_duplicates("patient_id")
    \
    report = {
        "source": "MIMIC-IV-ECG v1.0 machine_measurements.csv, open release",
        "mimic_ecgs": int(len(mimic)),
        "mimic_subjects": int(mimic["subject_id"].nunique()),
        "rate_sample": "one ECG per subject",
        "transferred": {
            "qrs_ge_120": "QRS duration is qrs_end minus qrs_onset, milliseconds, threshold 120.",
            "pr_ge_200": "PR interval is qrs_onset minus p_onset, milliseconds, threshold 200.",
            "axis_le_m30": "Global QRS axis, degrees, threshold -30.",
            "hr_ge_100": "Heart rate is 60000 divided by rr_interval, threshold 100 bpm.",
        },
        "withheld": {
            "sokolow_ge_3_5": "No lead-specific R or S amplitudes.",
            "st_ii_le_m0_1": "No lead II ST amplitude.",
            "q_ge_40": "No lead-specific Q-wave durations.",
            "qtc_long": "No Bazett QTc column, and sex is not in the open ECG table.",
        },
        "mimic_positive_rate": {
            "qrs_ge_120": rate(one["qrs_ms"] >= 120),
            "pr_ge_200": rate(one["pr_ms"] >= 200),
            "axis_le_m30": rate(one["axis"] <= -30),
            "hr_ge_100": rate(one["hr"] >= 100),
        },
        "ptbxl_test_positive_rate_12sl": {
            "qrs_ge_120": rate(ptb["QRS_Dur_Global"] >= 120),
            "pr_ge_200": rate(ptb["PR_Int_Global"] >= 200),
            "axis_le_m30": rate(ptb["R_AxisFrontal_Global"] <= -30),
            "hr_ge_100": rate(ptb["HR_Ventr_Global"] >= 100),
        },
    }
    \
\
    def positive(values: pd.Series, pred: pd.Series) -> dict:
        known = values.notna()
        return {
            "n_known": int(known.sum()),
            "n_missing": int((~known).sum()),
            "positive_rate": float(pred[known].mean()) if known.any() else None,
        }
    report["mimic_positive_rate"] = {
        "qrs_ge_120": positive(one["qrs_ms"], one["qrs_ms"] >= 120),
        "pr_ge_200": positive(one["pr_ms"], one["pr_ms"] >= 200),
        "axis_le_m30": positive(one["axis"], one["axis"] <= -30),
        "hr_ge_100": positive(one["hr"], one["hr"] >= 100),
    }
    report["ptbxl_test_positive_rate_12sl"] = {
        "qrs_ge_120": positive(ptb["QRS_Dur_Global"], ptb["QRS_Dur_Global"] >= 120),
        "pr_ge_200": positive(ptb["PR_Int_Global"], ptb["PR_Int_Global"] >= 200),
        "axis_le_m30": positive(ptb["R_AxisFrontal_Global"], ptb["R_AxisFrontal_Global"] <= -30),
        "hr_ge_100": positive(ptb["HR_Ventr_Global"], ptb["HR_Ventr_Global"] >= 100),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
