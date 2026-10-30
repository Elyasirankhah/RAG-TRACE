import json
import os
\
import numpy as np
import pandas as pd
import torch
from torch import nn
\
from noise_robustness import load_wave
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
WEIGHTS = os.path.join(OUT, "criteria_net.pt")
REPORT = os.path.join(OUT, "criteria_net.json")
HEADS = ["qrs_ge_120", "pr_ge_200", "axis_le_m30", "hr_ge_100"]
\
\
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, len(HEADS))
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(batch).squeeze(-1))
def downsample(sig: np.ndarray) -> np.ndarray:
    sig = np.nan_to_num(sig)[::5]
    if len(sig) < 1000:
        sig = np.pad(sig, ((0, 1000 - len(sig)), (0, 0)))
    return sig[:1000].T.astype(np.float32)
def labels_for(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["QRS_Dur_Global"].to_numpy() >= 120,
            frame["PR_Int_Global"].to_numpy() >= 200,
            frame["R_AxisFrontal_Global"].to_numpy() <= -30,
            frame["HR_Ventr_Global"].to_numpy() >= 100,
        ]
    ).astype(np.float32)
def load_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    waves = []
    kept = []
    for position, row in enumerate(frame.itertuples(index=False)):
        wave = downsample(load_wave(row.filename_hr))
        if not np.isfinite(wave).all():
            continue
        waves.append(wave)
        kept.append(position)
    usable = frame.iloc[kept]
    return np.stack(waves), labels_for(usable)
def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    \
    scores = []
    for column in range(y_true.shape[1]):
        if len(np.unique(y_true[:, column])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, column], y_score[:, column]))
    return float(np.mean(scores)) if scores else float("nan")
def predict(model: nn.Module, waves: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(waves), 64):
            batch = torch.from_numpy(waves[start : start + 64]).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs)
def main() -> None:
    cases = pd.read_csv(os.path.join(OUT, "case_record.csv"), usecols=["ecg_id", "patient_id", "split"])
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "filename_hr"])
    features = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "12sl_features.csv"),
        usecols=["ecg_id", "QRS_Dur_Global", "PR_Int_Global", "R_AxisFrontal_Global", "HR_Ventr_Global"],
    )
    data = cases.merge(database, on="ecg_id").merge(features, on="ecg_id")
    data = data.dropna(subset=["QRS_Dur_Global", "PR_Int_Global", "R_AxisFrontal_Global", "HR_Ventr_Global"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {}
    targets = {}
    for name in ("train", "calibrate", "test"):
        waves, target = load_split(data[data["split"] == name].reset_index(drop=True))
        splits[name] = waves
        targets[name] = target
        print(name, len(waves), flush=True)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_score = -1.0
    wait = 0
    history = []
    generator = torch.Generator().manual_seed(0)
    for epoch in range(1, 13):
        model.train()
        order = torch.randperm(len(splits["train"]), generator=generator).numpy()
        total = 0.0
        seen = 0
        for start in range(0, len(order), 64):
            index = order[start : start + 64]
            batch = torch.from_numpy(splits["train"][index]).to(device)
            label = torch.from_numpy(targets["train"][index]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), label)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        score = auroc(targets["calibrate"], predict(model, splits["calibrate"], device))
        history.append({"epoch": epoch, "train_loss": total / seen, "calibrate_auroc": score})
        print(history[-1], flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 3:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, WEIGHTS)
    test_score = predict(model, splits["test"], device)
    per_head = {}
    from sklearn.metrics import average_precision_score, roc_auc_score
    \
    for column, name in enumerate(HEADS):
        per_head[name] = {
            "auroc": float(roc_auc_score(targets["test"][:, column], test_score[:, column])),
            "auprc": float(average_precision_score(targets["test"][:, column], test_score[:, column])),
            "prevalence": float(targets["test"][:, column].mean()),
        }
    report = {
        "device": str(device),
        "train_ecgs": int(len(splits["train"])),
        "calibrate_ecgs": int(len(splits["calibrate"])),
        "test_ecgs": int(len(splits["test"])),
        "best_calibrate_macro_auroc": best_score,
        "test_macro_auroc": auroc(targets["test"], test_score),
        "per_criterion": per_head,
        "history": history,
        "input": "12 leads, 100 Hz, 10 seconds, from the PTB-XL 500 Hz waveform.",
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
