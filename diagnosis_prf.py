import json
import os

import numpy as np
import pandas as pd

from diagnosis_rule import CLASSES, affirmed_labels
from exploratory_revision import as_claims, borrow_no_rag
from rescore_corrected import bootstrap
from retrieval_harm import load_cases
from revision_evals import load_jsonl

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "diagnosis_prf.json")


def predicted(claims, extractor=affirmed_labels) -> set[str]:
    found = set()
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("type") != "diagnosis":
            continue
        found.update(extractor(claim))
    return found


def gold_of(row) -> set[str]:
    return {name for name in CLASSES if int(row[f"y_{name}"]) == 1}


def f1(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def arm_metrics(rows: list[dict], name: str) -> dict:
    pred_key = f"{name}_pred"
    gold_key = "gold"
    tp = fp = fn = 0
    per = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CLASSES}
    empty_pred = 0
    any_false = 0
    any_omission = 0
    exact = 0
    n = len(rows)
    for row in rows:
        pred = set(row[pred_key])
        gold = set(row[gold_key])
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        empty_pred += int(len(pred) == 0)
        any_false += int(bool(pred - gold))
        any_omission += int(bool(gold - pred))
        exact += int(pred == gold)
        for label in CLASSES:
            per[label]["tp"] += int(label in pred and label in gold)
            per[label]["fp"] += int(label in pred and label not in gold)
            per[label]["fn"] += int(label not in pred and label in gold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    by_class = {}
    for label, counts in per.items():
        class_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
        class_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
        by_class[label] = {
            "support": counts["tp"] + counts["fn"],
            "precision": class_precision,
            "recall": class_recall,
        }
    return {
        "n": n,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1(precision, recall),
        "true_positives": tp,
        "false_positives": fp,
        "omissions": fn,
        "mean_predicted": float(np.mean([len(row[pred_key]) for row in rows])),
        "mean_gold": float(np.mean([len(row[gold_key]) for row in rows])),
        "mean_omissions": fn / n,
        "notes_with_no_affirmed_diagnosis": empty_pred / n,
        "notes_with_any_false_diagnosis": any_false / n,
        "notes_with_any_omission": any_omission / n,
        "exact_set_match": exact / n,
        "by_class": by_class,
    }


def paired(rows: list[dict]) -> dict:
    if not rows or "no_retrieval_pred" not in rows[0] or "keyword_pred" not in rows[0]:
        return {}
    recall_a = []
    recall_b = []
    omit_a = []
    omit_b = []
    false_a = []
    false_b = []
    for row in rows:
        gold = set(row["gold"])
        if not gold:
            continue
        left = set(row["keyword_pred"])
        right = set(row["no_retrieval_pred"])
        recall_a.append(len(left & gold) / len(gold))
        recall_b.append(len(right & gold) / len(gold))
        omit_a.append(len(gold - left))
        omit_b.append(len(gold - right))
        false_a.append(int(bool(left - gold)))
        false_b.append(int(bool(right - gold)))
    recall_a = np.asarray(recall_a)
    recall_b = np.asarray(recall_b)
    rng = np.random.default_rng(2)
    draws = []
    n = len(recall_a)
    for _ in range(10000):
        index = rng.integers(0, n, n)
        draws.append(float(recall_a[index].mean() - recall_b[index].mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "patient_recall_keyword_minus_no_retrieval": {
            "difference": float(recall_a.mean() - recall_b.mean()),
            "ci95": [float(low), float(high)],
            "keyword": float(recall_a.mean()),
            "no_retrieval": float(recall_b.mean()),
            "n": int(n),
        },
        "mean_omissions_keyword_minus_no_retrieval": float(np.mean(omit_a) - np.mean(omit_b)),
        "any_false_keyword_minus_no_retrieval": bootstrap(np.asarray(false_a), np.asarray(false_b)),
    }


def score_notes(notes: dict, by_id, reference: dict | None = None, extractor=affirmed_labels) -> dict:
    borrowed = 0
    if reference is not None:
        notes, borrowed = borrow_no_rag(notes, reference)
    rows = []
    skipped = 0
    for ecg_id, record in notes.items():
        if ecg_id not in by_id:
            skipped += 1
            continue
        item = {"ecg_id": ecg_id, "gold": sorted(gold_of(by_id[ecg_id]))}
        usable = False
        for name, key in (("no_retrieval", "no_rag"), ("keyword", "with_keyword"), ("tfidf", "with_tfidf")):
            if key not in record:
                continue
            parsed = as_claims(record.get(key))
            if parsed is None:
                item = None
                break
            item[f"{name}_pred"] = sorted(predicted(parsed, extractor))
            usable = True
        if item is None or not usable or "keyword_pred" not in item:
            skipped += 1
            continue
        rows.append(item)
    summary = {"scored": len(rows), "skipped": skipped}
    if borrowed:
        summary["no_retrieval_source"] = "original no-card notes joined by electrocardiogram id; prompts were not matched"
        summary["borrowed_no_rag"] = borrowed
    if not rows:
        return summary
    for name in ("no_retrieval", "keyword", "tfidf"):
        if f"{name}_pred" not in rows[0]:
            continue
        summary[name] = arm_metrics(rows, name)
    summary["paired"] = paired(rows)
    return summary


def jobs() -> dict:
    heldout = load_jsonl(os.path.join(OUT, "heldout_generations.jsonl"))
    glm_heldout_path = os.path.join(OUT, "glm", "heldout_generations.jsonl")
    glm_heldout = load_jsonl(glm_heldout_path) if os.path.exists(glm_heldout_path) else None
    return {
        "gpt_heldout": (os.path.join(OUT, "heldout_generations.jsonl"), None),
        "gpt_numeric_cards": (os.path.join(OUT, "clean_generations.jsonl"), heldout),
        "gpt_matched_prompt": (os.path.join(OUT, "fair_generations.jsonl"), None),
        "gpt_constrained": (os.path.join(OUT, "constrained_generations.jsonl"), heldout),
        "gpt_matched_constrained": (os.path.join(OUT, "matched_constrained_generations.jsonl"), None),
        "gpt_matched_constrained": (os.path.join(OUT, "matched_constrained_generations.jsonl"), None),
        "qwen3_8b": (os.path.join(OUT, "open_models", "Qwen__Qwen3-8B.jsonl"), None),
        "qwen3_14b": (os.path.join(OUT, "open_models", "Qwen__Qwen3-14B.jsonl"), None),
        "gemma3_12b": (os.path.join(OUT, "open_models", "google__gemma-3-12b-it.jsonl"), None),
        "medgemma_4b": (os.path.join(OUT, "open_models", "google__medgemma-4b-it.jsonl"), None),
        "deepseek_r1_8b": (os.path.join(OUT, "open_models", "deepseek-ai__DeepSeek-R1-Distill-Llama-8B.jsonl"), None),
        "glm_heldout": (glm_heldout_path, None),
        "glm_numeric_cards": (os.path.join(OUT, "glm", "clean_generations.jsonl"), glm_heldout),
        "glm_matched_prompt": (os.path.join(OUT, "glm", "fair_generations.jsonl"), None),
        "glm_constrained": (os.path.join(OUT, "glm", "constrained_generations.jsonl"), glm_heldout),
    }


def main() -> None:
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    report = {
        "reference": "PTB-XL superclass codes. Agreement with a stored code, not a clinical reading of the sentence.",
        "prediction": "Deduplicated label-aware affirmed superclasses. The cue list is not used.",
        "arms": {},
    }
    brief = {}
    for name, (path, reference) in jobs().items():
        if not path or not os.path.exists(path):
            report["arms"][name] = {"missing": path}
            continue
        summary = score_notes(load_jsonl(path), by_id, reference)
        report["arms"][name] = summary
        keyword = summary.get("keyword", {})
        paired_recall = summary.get("paired", {}).get("patient_recall_keyword_minus_no_retrieval")
        brief[name] = {
            "n": summary.get("scored"),
            "keyword_precision": keyword.get("micro_precision"),
            "keyword_recall": keyword.get("micro_recall"),
            "keyword_f1": keyword.get("micro_f1"),
            "keyword_omissions": keyword.get("omissions"),
            "keyword_any_false": keyword.get("notes_with_any_false_diagnosis"),
            "keyword_any_omission": keyword.get("notes_with_any_omission"),
            "recall_change": None if not paired_recall else paired_recall["difference"],
            "recall_ci": None if not paired_recall else paired_recall["ci95"],
        }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
