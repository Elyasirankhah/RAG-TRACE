import json
import os

import numpy as np

from diagnosis_rule import affirmed_labels
from exploratory_revision import as_claims
from retrieval_harm import load_cases
from revision_evals import load_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "micro_f1_forest.json")
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def gold_of(row) -> set[str]:
    return {name for name in CLASSES if int(row[f"y_{name}"]) == 1}


def predicted(claims) -> set[str]:
    out = set()
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("type") != "diagnosis":
            continue
        out.update(affirmed_labels(claim))
    return out


def counts(pred: set[str], gold: set[str]) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def f1_from(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def percentile_interval(draws: np.ndarray) -> list[float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def score(path: str, label: str) -> dict:
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    notes = load_jsonl(path)
    rows = []
    skipped = 0
    for ecg_id, record in notes.items():
        if ecg_id not in by_id:
            skipped += 1
            continue
        left = as_claims(record.get("no_rag"))
        right = as_claims(record.get("with_keyword"))
        if left is None or right is None:
            skipped += 1
            continue
        gold = gold_of(by_id[ecg_id])
        rows.append((counts(predicted(left), gold), counts(predicted(right), gold)))
    if not rows:
        return {"label": label, "path": path, "scored": 0, "skipped": skipped}

    per_no = np.array([a for a, _ in rows], dtype=int)
    per_kw = np.array([b for _, b in rows], dtype=int)
    n = per_no.shape[0]

    no_sum = per_no.sum(axis=0)
    kw_sum = per_kw.sum(axis=0)
    no_f1 = f1_from(int(no_sum[0]), int(no_sum[1]), int(no_sum[2]))
    kw_f1 = f1_from(int(kw_sum[0]), int(kw_sum[1]), int(kw_sum[2]))

    rng = np.random.default_rng(2)
    draws = np.empty(10000)
    for i in range(draws.shape[0]):
        index = rng.integers(0, n, n)
        a = per_no[index].sum(axis=0)
        b = per_kw[index].sum(axis=0)
        draws[i] = f1_from(int(b[0]), int(b[1]), int(b[2])) - f1_from(int(a[0]), int(a[1]), int(a[2]))

    return {
        "label": label,
        "path": os.path.relpath(path, ROOT),
        "scored": int(n),
        "skipped": int(skipped),
        "micro_f1": {
            "no_retrieval": float(no_f1),
            "keyword": float(kw_f1),
            "difference": float(kw_f1 - no_f1),
            "ci95": percentile_interval(draws),
        },
    }


def main() -> None:
    jobs = [
        ("gpt-4.1-mini", os.path.join(OUT, "heldout_generations.jsonl")),
        ("GLM-5.3-Flash", os.path.join(OUT, "glm", "heldout_generations.jsonl")),
        ("Qwen3-8B", os.path.join(OUT, "open_models", "Qwen__Qwen3-8B.jsonl")),
        ("Qwen3-14B", os.path.join(OUT, "open_models", "Qwen__Qwen3-14B.jsonl")),
        ("Gemma 3 12B", os.path.join(OUT, "open_models", "google__gemma-3-12b-it.jsonl")),
        ("MedGemma 4B", os.path.join(OUT, "open_models", "google__medgemma-4b-it.jsonl")),
    ]
    report = {
        "reference": "PTB-XL superclass codes. Agreement with a stored code, not a clinical reading.",
        "bootstrap": "10,000 patient resamples, seed 2. Difference is keyword minus no retrieval.",
        "models": [],
    }
    for label, path in jobs:
        if not os.path.exists(path):
            report["models"].append({"label": label, "missing": os.path.relpath(path, ROOT)})
            continue
        report["models"].append(score(path, label))
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

