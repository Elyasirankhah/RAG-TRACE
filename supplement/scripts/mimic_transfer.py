import json
import os
import urllib.request
\
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
\
from train_criteria_net import HEADS, WEIGHTS, Net, downsample
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "mimic_transfer.json")
WAVE = os.path.join(ROOT, "mimic-iv-ecg", "waves")
BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"
N = 200
\
\
def file_name(subject_id: int, study_id: int) -> str:
    subject = str(int(subject_id))
    study = str(int(study_id))
    return f"files/p{subject[:4]}/p{subject}/s{study}/{study}"
def download_record(name: str) -> str:
    os.makedirs(os.path.join(WAVE, os.path.dirname(name)), exist_ok=True)
    stem = os.path.join(WAVE, name)
    for suffix in (".hea", ".dat"):
        path = stem + suffix
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + name + suffix, path)
    return stem
def read_mimic(stem: str) -> np.ndarray:
    header = open(stem + ".hea", encoding="utf-8").read().splitlines()
    n_sig = int(header[0].split()[1])
    n_samples = int(header[0].split()[3])
    raw = np.fromfile(stem + ".dat", dtype="<i2").reshape(n_samples, n_sig).astype(float)
    gains = []
    for line in header[1 : 1 + n_sig]:
        gain = float(line.split()[2].split("(")[0].split("/")[0])
        gains.append(1.0 if gain == 0 else gain)
    return raw / np.array(gains)
def machine_labels(frame: pd.DataFrame) -> np.ndarray:
    qrs = frame["qrs_end"] - frame["qrs_onset"]
    pr = frame["qrs_onset"] - frame["p_onset"]
    hr = 60_000 / frame["rr_interval"]
    return np.column_stack(
        [
            qrs.to_numpy() >= 120,
            pr.to_numpy() >= 200,
            frame["qrs_axis"].to_numpy() <= -30,
            hr.to_numpy() >= 100,
        ]
    ).astype(float)
def main() -> None:
    if not os.path.exists(WEIGHTS):
        raise SystemExit("criteria network is not trained yet")
    measurements = pd.read_csv(
        os.path.join(ROOT, "mimic-iv-ecg", "machine_measurements.csv"),
        usecols=["subject_id", "study_id", "rr_interval", "p_onset", "qrs_onset", "qrs_end", "qrs_axis"],
    )
    merged = measurements.dropna()
    merged = merged[(merged["rr_interval"] > 0) & (merged["qrs_end"] > merged["qrs_onset"])]
    sample = merged.sample(n=N * 2, random_state=21)
    waves = []
    study_ids = []
    for row in sample.itertuples(index=False):
        if len(waves) >= N:
            break
        try:
            stem = download_record(file_name(row.subject_id, row.study_id))
            wave = downsample(read_mimic(stem))
        except Exception as exc:
            print(f"skip {row.study_id}: {type(exc).__name__}", flush=True)
            continue
        if not np.isfinite(wave).all():
            continue
        waves.append(wave)
        study_ids.append(int(row.study_id))
        print(f"wave {len(waves)}", flush=True)
    usable = sample[sample["study_id"].isin(study_ids)].copy()
    usable["_order"] = usable["study_id"].map({study_id: index for index, study_id in enumerate(study_ids)})
    usable = usable.sort_values("_order")
    target = machine_labels(usable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    stacked = np.stack(waves)
    scores = []
    with torch.no_grad():
        for start in range(0, len(stacked), 64):
            batch = torch.from_numpy(stacked[start : start + 64]).to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    score = np.concatenate(scores)
    per_head = {}
    for column, name in enumerate(HEADS):
        known = np.isfinite(target[:, column])
        per_head[name] = {
            "n": int(known.sum()),
            "prevalence": float(np.nanmean(target[:, column])),
            "auroc": None if known.sum() < 5 or len(np.unique(target[known, column])) < 2 else float(roc_auc_score(target[known, column], score[known, column])),
            "auprc": None if known.sum() < 5 or len(np.unique(target[known, column])) < 2 else float(average_precision_score(target[known, column], score[known, column])),
        }
    report = {"studies": int(len(usable)), "per_criterion": per_head}
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
