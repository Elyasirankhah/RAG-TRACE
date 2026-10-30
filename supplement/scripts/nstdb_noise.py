import json
import os
import urllib.request
\
import numpy as np
import pandas as pd
from scipy import signal
\
from noise_robustness import CHECKS, FS, extract, holds_from, load_wave, rate_block, withhold_map
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
NOISE_DIR = os.path.join(ROOT, "nstdb")
REPORT = os.path.join(OUT, "nstdb_noise.json")
BASE = "https://physionet.org/files/nstdb/1.0.0/"
RECORDS = ["bw", "em", "ma"]
\
\
def download() -> None:
    os.makedirs(NOISE_DIR, exist_ok=True)
    for name in RECORDS:
        for suffix in (".hea", ".dat"):
            path = os.path.join(NOISE_DIR, name + suffix)
            if not os.path.exists(path):
                urllib.request.urlretrieve(BASE + name + suffix, path)
def read_212(path: str, n_samples: int, n_sig: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3)
    first = raw[:, 0].astype(np.int32) + ((raw[:, 1].astype(np.int32) & 0x0F) << 8)
    second = raw[:, 2].astype(np.int32) + ((raw[:, 1].astype(np.int32) & 0xF0) << 4)
    samples = np.empty(first.size + second.size, dtype=np.int32)
    samples[0::2] = first
    samples[1::2] = second
    samples = samples.astype(np.int32)
    samples[samples >= 2048] -= 4096
    usable = (len(samples) // n_sig) * n_sig
    return samples[:usable].reshape(-1, n_sig)[:n_samples]
def load_noise(name: str) -> np.ndarray:
    header = open(os.path.join(NOISE_DIR, name + ".hea"), encoding="utf-8").read().splitlines()
    n_sig = int(header[0].split()[1])
    source_fs = float(header[0].split()[2])
    n_samples = int(header[0].split()[3])
    gain = float(header[1].split()[2].split("(")[0].split("/")[0])
    if gain == 0:
        gain = 1.0
    values = read_212(os.path.join(NOISE_DIR, name + ".dat"), n_samples, n_sig) / gain
    target = int(len(values) * FS / source_fs)
    resampled = signal.resample(values, target, axis=0)
    return resampled.astype(float)
def mix(clean: np.ndarray, noise: np.ndarray, level: float, offset: int) -> np.ndarray:
    take = len(clean)
    start = offset % (len(noise) - take)
    clip = noise[start : start + take]
    if clip.shape[1] == 1:
        clip = np.repeat(clip, 2, axis=1)
    pattern = np.zeros_like(clean)
    pattern[:, :6] = clip[:, 0:1]
    pattern[:, 6:] = clip[:, 1:2]
    signal_rms = np.sqrt(np.mean(clean**2, axis=0)) + 1e-8
    noise_rms = np.sqrt(np.mean(pattern**2, axis=0)) + 1e-8
    return clean + pattern * (level * signal_rms / noise_rms)
def main() -> None:
    download()
    noises = {name: load_noise(name) for name in RECORDS}
    for name, values in noises.items():
        print(name, values.shape, float(np.nanmean(np.abs(values))), flush=True)
    gates = json.load(open(os.path.join(OUT, "noise_robustness.json"), encoding="utf-8"))[
        "quality_gates_99th_percentile_of_clean_calibration"
    ]
    sample = pd.read_csv(os.path.join(OUT, "noise_sample.csv"))
    if "filename_hr" not in sample.columns:
        database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "filename_hr"])
        sample = sample.merge(database, on="ecg_id")
    clean_cache = {}
    for _, row in sample.iterrows():
        sig = load_wave(row["filename_hr"])
        meas, _ = extract(sig)
        clean_cache[int(row["ecg_id"])] = (sig, holds_from(meas))
    curves = []
    for name, noise in noises.items():
        for level in (0.1, 0.25, 0.5):
            decisions = []
            for ecg_id, (clean, clean_holds) in clean_cache.items():
                mixed = mix(clean, noise, level, offset=ecg_id * 500)
                meas, ratios = extract(mixed)
                naive = holds_from(meas)
                present = np.ones(12, dtype=bool)
                blocked = withhold_map(ratios, gates, present)
                for check in CHECKS:
                    guard = None if blocked[check] else naive[check]
                    decisions.append({"clean": clean_holds[check], "naive": naive[check], "guard": guard})
            curves.append({"noise": name, "level": level, **rate_block(decisions)})
    report = {
        "source": "MIT-BIH Noise Stress Test Database, records bw, em, and ma",
        "evaluation_ecgs": len(clean_cache),
        "level_means": "Noise RMS divided by the clean-lead RMS.",
        "gates_reused_from": "noise_robustness.json",
        "curves": curves,
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
