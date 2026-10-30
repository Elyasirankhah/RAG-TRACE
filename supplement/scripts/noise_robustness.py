import json
import os
\
import numpy as np
import pandas as pd
from scipy import signal
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "noise_robustness.json")
FS = 500
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
N_EVAL = 100
N_CAL = 80
CHECKS = ["qrs_ge_120", "axis_le_m30", "sokolow_ge_3_5", "st_ii_le_m0_1", "q_ge_40", "hr_ge_100"]
\
\
def load_wave(filename_hr: str) -> np.ndarray:
    path = os.path.join(ROOT, "ptb-xl", filename_hr)
    header = open(path + ".hea", encoding="utf-8").read().splitlines()
    n_sig = int(header[0].split()[1])
    n_samples = int(header[0].split()[3])
    raw = np.fromfile(path + ".dat", dtype="<i2").reshape(n_samples, n_sig).astype(float)
    gains = []
    for line in header[1 : 1 + n_sig]:
        token = line.split()[2]
        gains.append(float(token.split("(")[0].split("/")[0]))
    return raw / np.array(gains)
def band_rms(x: np.ndarray, low: float, high: float) -> float:
    if low <= 0:
        sos = signal.butter(2, high, btype="lowpass", fs=FS, output="sos")
    else:
        sos = signal.butter(2, [low, high], btype="bandpass", fs=FS, output="sos")
    y = signal.sosfiltfilt(sos, np.nan_to_num(x))
    return float(np.sqrt(np.mean(y**2)))
def qrs_times(lead: np.ndarray) -> np.ndarray:
    sos = signal.butter(2, [5, 15], btype="bandpass", fs=FS, output="sos")
    filtered = signal.sosfiltfilt(sos, np.nan_to_num(lead))
    deriv = np.diff(filtered, prepend=filtered[0])
    window = int(0.15 * FS)
    integrated = np.convolve(deriv * deriv, np.ones(window) / window, mode="same")
    height = np.percentile(integrated, 97) * 0.35
    peaks, _ = signal.find_peaks(integrated, distance=int(0.3 * FS), height=height)
    morph = signal.sosfiltfilt(signal.butter(2, [0.5, 40], btype="bandpass", fs=FS, output="sos"), np.nan_to_num(lead))
    refined = []
    for peak in peaks:
        start = max(0, peak - int(0.08 * FS))
        stop = min(len(morph), peak + int(0.08 * FS))
        refined.append(start + int(np.argmax(np.abs(morph[start:stop]))))
    return np.array(refined, dtype=int), morph
def qrs_bounds(morph: np.ndarray, r_index: int) -> tuple[int, int]:
    deriv = np.abs(np.diff(morph, prepend=morph[0]))
    left = max(0, r_index - int(0.12 * FS))
    right = min(len(morph) - 1, r_index + int(0.16 * FS))
    scale = np.max(deriv[left:right]) + 1e-8
    onset = left
    for index in range(r_index, left, -1):
        if deriv[index] < 0.1 * scale and abs(morph[index]) < 0.2 * max(abs(morph[r_index]), 0.1):
            onset = index
            break
    offset = right
    for index in range(r_index, right):
        if deriv[index] < 0.1 * scale and abs(morph[index]) < 0.2 * max(abs(morph[r_index]), 0.1):
            offset = index
            break
    return onset, max(offset, onset + 1)
def extract(sig: np.ndarray) -> dict:
    present = ~np.all(np.isnan(sig), axis=0)
    rhythm_lead = 1 if present[1] else (0 if present[0] else None)
    result = {name: None for name in ["hr", "qrs_ms", "q_ms", "st_mv", "sokolow", "axis"]}
    ratios = {"baseline": None, "powerline": None, "muscle": None}
    if rhythm_lead is None:
        return result, ratios
    lead = sig[:, rhythm_lead]
    ratios = {
        "baseline": band_rms(lead, 0.0, 0.5) / (band_rms(lead, 0.5, 40) + 1e-6),
        "powerline": band_rms(lead, 48, 52) / (band_rms(lead, 0.5, 40) + 1e-6),
        "muscle": band_rms(lead, 40, 100) / (band_rms(lead, 5, 15) + 1e-6),
    }
    r_times, morph = qrs_times(lead)
    if len(r_times) < 4:
        return result, ratios
    rr = np.diff(r_times) / FS
    result["hr"] = float(60 / np.median(rr))
    durations, q_durations, st_values, axes = [], [], [], []
    s_v1, r_v5, r_v6 = [], [], []
    for r_index in r_times[1:-1]:
        onset, offset = qrs_bounds(morph, r_index)
        durations.append((offset - onset) * 1000 / FS)
        segment = morph[onset:r_index]
        if len(segment) and np.min(segment) <= -0.05:
            q_durations.append((np.argmin(segment) + 1) * 1000 / FS)
        else:
            q_durations.append(0.0)
        st_at = min(len(morph) - 1, offset + int(0.08 * FS))
        baseline_start = max(0, onset - int(0.06 * FS))
        st_values.append(float(morph[st_at] - np.median(morph[baseline_start:onset])))
        if present[0] and present[5]:
            area_i = np.sum(sig[onset:offset, 0])
            area_f = np.sum(sig[onset:offset, 5])
            axes.append(float(np.degrees(np.arctan2(area_f, area_i))))
        if present[6]:
            window = sig[r_index : min(len(sig), r_index + int(0.08 * FS)), 6]
            if len(window):
                s_v1.append(float(np.min(window)))
        if present[10]:
            window = sig[max(0, r_index - int(0.04 * FS)) : r_index + int(0.04 * FS), 10]
            if len(window):
                r_v5.append(float(np.max(window)))
        if present[11]:
            window = sig[max(0, r_index - int(0.04 * FS)) : r_index + int(0.04 * FS), 11]
            if len(window):
                r_v6.append(float(np.max(window)))
    result["qrs_ms"] = float(np.median(durations))
    result["q_ms"] = float(np.median(q_durations))
    result["st_mv"] = float(np.median(st_values))
    if axes:
        result["axis"] = float(np.median(axes))
    if s_v1 and r_v5 and r_v6:
        result["sokolow"] = float(abs(np.median(s_v1)) + max(np.median(r_v5), np.median(r_v6)))
    return result, ratios
def holds_from(meas: dict) -> dict:
    return {
        "qrs_ge_120": None if meas["qrs_ms"] is None else meas["qrs_ms"] >= 120,
        "axis_le_m30": None if meas["axis"] is None else meas["axis"] <= -30,
        "sokolow_ge_3_5": None if meas["sokolow"] is None else meas["sokolow"] >= 3.5,
        "st_ii_le_m0_1": None if meas["st_mv"] is None else meas["st_mv"] <= -0.1,
        "q_ge_40": None if meas["q_ms"] is None else meas["q_ms"] >= 40,
        "hr_ge_100": None if meas["hr"] is None else meas["hr"] >= 100,
    }
def withhold_map(ratios: dict, gates: dict, present: np.ndarray) -> dict:
    blocked = {check: False for check in CHECKS}
    if ratios["baseline"] is not None and ratios["baseline"] > gates["baseline"]:
        blocked["st_ii_le_m0_1"] = True
    if ratios["powerline"] is not None and ratios["powerline"] > gates["powerline"]:
        blocked["sokolow_ge_3_5"] = True
        blocked["st_ii_le_m0_1"] = True
        blocked["q_ge_40"] = True
    if ratios["muscle"] is not None and ratios["muscle"] > gates["muscle"]:
        blocked["qrs_ge_120"] = True
        blocked["sokolow_ge_3_5"] = True
        blocked["st_ii_le_m0_1"] = True
        blocked["q_ge_40"] = True
    if not present[1]:
        blocked["qrs_ge_120"] = True
        blocked["st_ii_le_m0_1"] = True
        blocked["q_ge_40"] = True
        blocked["hr_ge_100"] = True
    if not (present[0] and present[5]):
        blocked["axis_le_m30"] = True
    if not (present[6] and present[10] and present[11]):
        blocked["sokolow_ge_3_5"] = True
    return blocked
def degrade(clean: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sig = clean.copy()
    present = np.ones(12, dtype=bool)
    n = len(sig)
    t = np.arange(n) / FS
    if kind == "baseline":
        phase = rng.uniform(0, 2 * np.pi, 12)
        sig += level * np.sin(2 * np.pi * 0.33 * t[:, None] + phase)
    elif kind == "powerline":
        phase = rng.uniform(0, 2 * np.pi, 12)
        sig += level * np.sin(2 * np.pi * 50 * t[:, None] + phase)
    elif kind == "muscle":
        noise = rng.normal(0, 1, sig.shape)
        sos = signal.butter(2, [20, 80], btype="bandpass", fs=FS, output="sos")
        shaped = signal.sosfiltfilt(sos, noise, axis=0)
        current = np.sqrt(np.mean(shaped**2, axis=0)) + 1e-8
        sig += shaped * (level / current)
    elif kind == "missing":
        groups = {
            1: [6],
            2: [1],
            3: [0, 5],
            4: [6, 7, 8, 9, 10, 11],
        }
        for index in groups[int(level)]:
            sig[:, index] = np.nan
            present[index] = False
    return sig, present
def choose_ids(db: pd.DataFrame, split: str, blocked: set[int], count: int, seed: int) -> pd.DataFrame:
    pool = db[(db["split"] == split) & ~db["patient_id"].astype(int).isin(blocked)]
    rng = np.random.default_rng(seed)
    patients = pool["patient_id"].astype(int).drop_duplicates().to_numpy()
    picked = set(rng.choice(patients, size=count, replace=False).tolist())
    rows = []
    for patient_id, group in pool.groupby(pool["patient_id"].astype(int)):
        if patient_id in picked:
            rows.append(group.sample(n=1, random_state=int(rng.integers(0, 10_000))))
    return pd.concat(rows, ignore_index=True)
def rate_block(decisions: list[dict]) -> dict:
    naive_false = guard_false = naive_miss = guard_said_false = known = true_ref = withheld = 0
    for item in decisions:
        if item["clean"] is None:
            continue
        known += 1
        true_ref += int(item["clean"])
        naive_false += int(item["naive"] is True and item["clean"] is False)
        guard_false += int(item["guard"] is True and item["clean"] is False)
        naive_miss += int(item["naive"] is False and item["clean"] is True)
        guard_said_false += int(item["guard"] is False and item["clean"] is True)
        withheld += int(item["guard"] is None)
    return {
        "n_criteria": known,
        "naive_false_release": naive_false / known,
        "guard_false_release": guard_false / known,
        "naive_said_absent_when_clean_true": naive_miss / true_ref if true_ref else None,
        "guard_said_absent_when_clean_true": guard_said_false / true_ref if true_ref else None,
        "guard_withhold_rate": withheld / known,
    }
def main() -> None:
    cases = pd.read_csv(os.path.join(OUT, "case_record.csv"), usecols=["ecg_id", "patient_id", "split"])
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "filename_hr"])
    db = cases.merge(database, on="ecg_id")
    blocked = set(pd.read_csv(os.path.join(OUT, "annotation_sample.csv"))["patient_id"].astype(int))
    blocked.update(pd.read_csv(os.path.join(OUT, "confirm_sample.csv"))["patient_id"].astype(int))
    evaluation = choose_ids(db, "test", blocked, N_EVAL, 3)
    calibration = choose_ids(db, "calibrate", blocked, N_CAL, 4)
    evaluation.to_csv(os.path.join(OUT, "noise_sample.csv"), index=False)
    \
    cal_ratios = []
    for filename in calibration["filename_hr"]:
        _, ratios = extract(load_wave(filename))
        cal_ratios.append(ratios)
    cal = pd.DataFrame(cal_ratios)
    gates = {name: float(np.nanpercentile(cal[name], 99)) for name in ["baseline", "powerline", "muscle"]}
    \
    sl = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "12sl_features.csv"),
        usecols=["ecg_id", "QRS_Dur_Global", "HR_Ventr_Global", "R_AxisFrontal_Global", "S_Amp_V1", "R_Amp_V5", "R_Amp_V6", "ST_Amp_II", "Q_Dur_II"],
    ).set_index("ecg_id")
    \
    clean_cache = {}
    agreement = {check: {"agree": 0, "known": 0} for check in CHECKS}
    for _, row in evaluation.iterrows():
        sig = load_wave(row["filename_hr"])
        meas, ratios = extract(sig)
        clean_cache[int(row["ecg_id"])] = (sig, meas, holds_from(meas))
        reference = sl.loc[int(row["ecg_id"])]
        sl_holds = {
            "qrs_ge_120": float(reference["QRS_Dur_Global"]) >= 120,
            "axis_le_m30": float(reference["R_AxisFrontal_Global"]) <= -30,
            "sokolow_ge_3_5": abs(float(reference["S_Amp_V1"])) + max(float(reference["R_Amp_V5"]), float(reference["R_Amp_V6"])) >= 3.5,
            "st_ii_le_m0_1": float(reference["ST_Amp_II"]) <= -0.1,
            "q_ge_40": float(reference["Q_Dur_II"]) >= 40,
            "hr_ge_100": float(reference["HR_Ventr_Global"]) >= 100,
        }
        wave_holds = holds_from(meas)
        for check in CHECKS:
            if wave_holds[check] is None or pd.isna(reference["QRS_Dur_Global"]):
                continue
            agreement[check]["known"] += 1
            agreement[check]["agree"] += int(wave_holds[check] == sl_holds[check])
    conditions = [("clean", 0.0)]
    conditions += [("baseline", level) for level in (0.15, 0.5, 1.0)]
    conditions += [("muscle", level) for level in (0.05, 0.15, 0.4)]
    conditions += [("powerline", level) for level in (0.05, 0.2, 0.5)]
    conditions += [("missing", level) for level in (1, 2, 3, 4)]
    \
    curves = []
    for kind, level in conditions:
        decisions = []
        rng = np.random.default_rng(10 + int(level * 100) + sum(ord(char) for char in kind))
        for ecg_id, (clean, _, clean_holds) in clean_cache.items():
            if kind == "clean":
                sig, present = clean, np.ones(12, dtype=bool)
            else:
                sig, present = degrade(clean, kind, level, rng)
            meas, ratios = extract(np.nan_to_num(sig))
            naive = holds_from(meas)
            _, guard_ratios = extract(sig)
            ratios = guard_ratios if kind == "missing" else ratios
            blocked_checks = withhold_map(ratios, gates, present)
            for check in CHECKS:
                guard = None if blocked_checks[check] else naive[check]
                decisions.append({"clean": clean_holds[check], "naive": naive[check], "guard": guard})
        curves.append({"noise": kind, "level": level, **rate_block(decisions)})
    report = {
        "evaluation_ecgs": N_EVAL,
        "calibration_ecgs": N_CAL,
        "withheld_without_wave_measurement": ["pr_ge_200", "qtc_long"],
        "quality_gates_99th_percentile_of_clean_calibration": gates,
        "clean_agreement_with_12sl": {
            check: (item["agree"] / item["known"] if item["known"] else None) for check, item in agreement.items()
        },
        "curves": curves,
        "reference": "Each noisy decision is compared with the same extractor on the clean waveform.",
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
