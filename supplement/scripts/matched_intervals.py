import json
import os
\
import numpy as np
\
from diagnosis_prf import gold_of, predicted
from exploratory_revision import as_claims
from retrieval_harm import load_cases
from revision_evals import load_jsonl
\
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "ptbxl_baseline", "matched_constrained_generations.jsonl")
OUT = os.path.join(ROOT, "results", "ptbxl_baseline", "matched_intervals.json")
\
\
def counts(pred: set[str], gold: set[str]) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)
def f1_from(tp, fp, fn) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
def percentile_interval(draws: np.ndarray) -> list[float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]
def main() -> None:
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    notes = load_jsonl(RAW)
    rows = []
    for ecg_id, record in notes.items():
        if ecg_id not in by_id:
            continue
        left = as_claims(record.get("no_rag"))
        right = as_claims(record.get("with_keyword"))
        if left is None or right is None:
            continue
        gold = gold_of(by_id[ecg_id])
        rows.append((predicted(left), predicted(right), gold))
    n = len(rows)
    no_tp = no_fp = no_fn = 0
    kw_tp = kw_fp = kw_fn = 0
    hyp_no = []
    hyp_kw = []
    for no_pred, kw_pred, gold in rows:
        a, b, c = counts(no_pred, gold)
        no_tp += a
        no_fp += b
        no_fn += c
        a, b, c = counts(kw_pred, gold)
        kw_tp += a
        kw_fp += b
        kw_fn += c
        if "HYP" in gold:
            hyp_no.append(int("HYP" in no_pred))
            hyp_kw.append(int("HYP" in kw_pred))
    no_f1 = f1_from(no_tp, no_fp, no_fn)
    kw_f1 = f1_from(kw_tp, kw_fp, kw_fn)
    hyp_no = np.asarray(hyp_no)
    hyp_kw = np.asarray(hyp_kw)
    no_hyp = float(hyp_no.mean())
    kw_hyp = float(hyp_kw.mean())
    \
    per_no = np.array([counts(no_pred, gold) for no_pred, _, gold in rows])
    per_kw = np.array([counts(kw_pred, gold) for _, kw_pred, gold in rows])
    rng = np.random.default_rng(2)
    f1_draws = np.empty(10000)
    for draw in range(10000):
        index = rng.integers(0, n, n)
        left = per_no[index].sum(axis=0)
        right = per_kw[index].sum(axis=0)
        f1_draws[draw] = f1_from(*right) - f1_from(*left)
    hyp_draws = np.empty(10000)
    m = len(hyp_no)
    for draw in range(10000):
        index = rng.integers(0, m, m)
        hyp_draws[draw] = float(hyp_kw[index].mean() - hyp_no[index].mean())
    report = {
        "n": n,
        "bootstrap": "10,000 patient resamples, seed 2. Difference is constrained minus no cards.",
        "hypertrophy_support": int(m),
        "micro_f1": {
            "no_cards": no_f1,
            "constrained": kw_f1,
            "difference": kw_f1 - no_f1,
            "ci95": percentile_interval(f1_draws),
        },
        "hypertrophy_recall": {
            "no_cards": no_hyp,
            "constrained": kw_hyp,
            "difference": kw_hyp - no_hyp,
            "ci95": percentile_interval(hyp_draws),
        },
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
