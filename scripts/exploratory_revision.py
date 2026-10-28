import json
import os

import pandas as pd

from claim_scoring import SCP_FOR_CHECK
from diagnosis_rule import affirmed_labels, text_only_labels
from ecg_guard import SCALAR, verify
from independent_measurements import measurements_for
from rescore_corrected import bootstrap
from retrieval_harm import facts_for, load_cases
from revision_evals import load_jsonl, scp_table, unig_table
from score_qwen import claims_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "exploratory_revision.json")
MARGINS = json.load(open(os.path.join(ROOT, "results", "ecg_guard", "development_report.json"), encoding="utf-8"))
MARGINS = MARGINS["margins_95th_abs_12sl_minus_unig"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

ORDINARY = [
    ("The ECG is normal.", True),
    ("The ECG is interpreted as normal.", True),
    ("The tracing is normal.", True),
    ("Diagnosis: myocardial infarction.", True),
    ("This ECG shows normal sinus rhythm.", True),
    ("Findings are diagnostic of conduction disturbance.", True),
    ("The recording demonstrates left ventricular hypertrophy.", True),
    ("The ECG shows nonspecific ST-T changes.", True),
    ("Infarction cannot be excluded.", False),
    ("This is not possible.", False),
    ("Infarction is not possible.", False),
    ("The pattern is not consistent with infarction.", False),
    ("A normal ECG is impossible here.", False),
    ("Rule out myocardial infarction.", False),
    ("No signs of hypertrophy.", False),
    ("Unlikely to be a bundle branch block.", False),
    ("The note does not show a Q wave.", False),
    ("Possible infarction, but there is no evidence of myocardial infarction.", False),
    ("The ECG is interpreted as normal sinus rhythm with no evidence of myocardial infarction.", True),
]


def check_ordinary() -> dict:
    rows = []
    for text, expected in ORDINARY:
        got = bool(affirmed_labels({"type": "diagnosis", "text": text, "label": None}))
        rows.append({"text": text, "expected": expected, "got": got, "match": got == expected})
    labeled = affirmed_labels({"type": "diagnosis", "text": "The ECG is normal.", "label": "NORM"})
    bare = affirmed_labels({"type": "diagnosis", "text": "NORM", "label": "NORM"})
    return {
        "n": len(rows),
        "matches": sum(row["match"] for row in rows),
        "failures": [row for row in rows if not row["match"]],
        "labeled_normal_counts": labeled == ["NORM"],
        "bare_code_with_label_counts": bare == ["NORM"],
    }


def as_claims(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return claims_of(raw)
    return None


def diagnosis_flags(claims, facts, labels) -> dict:
    absent = False
    low = False
    counted = 0
    for claim in claims:
        for label in affirmed_labels(claim):
            counted += 1
            if labels.get(label, 0) != 1:
                absent = True
            probability = facts["probabilities"].get(label)
            if probability is None or probability < 0.5:
                low = True
    return {"absent": int(absent), "low": int(low), "n": counted}


def score_records(notes: dict, by_id) -> dict:
    rows = []
    skipped = 0
    for ecg_id, record in notes.items():
        arms = {}
        for name, key in (("no_retrieval", "no_rag"), ("keyword", "with_keyword"), ("tfidf", "with_tfidf")):
            if key not in record:
                continue
            parsed = as_claims(record.get(key))
            if parsed is None:
                arms = {}
                break
            arms[name] = parsed
        if "keyword" not in arms:
            skipped += 1
            continue
        if ecg_id not in by_id:
            skipped += 1
            continue
        source = by_id[ecg_id]
        facts = facts_for(source)
        y = {name: int(source[f"y_{name}"]) for name in CLASSES}
        item = {"ecg_id": ecg_id}
        for name, claims in arms.items():
            flags = diagnosis_flags(claims, facts, y)
            item[f"{name}_absent"] = flags["absent"]
            item[f"{name}_low"] = flags["low"]
            item[f"{name}_n"] = flags["n"]
        rows.append(item)
    frame = pd.DataFrame(rows)
    summary = {"scored": int(len(frame)), "skipped": skipped}
    if frame.empty:
        return summary
    present = [name for name in ("no_retrieval", "keyword", "tfidf") if f"{name}_absent" in frame.columns]
    for name in present:
        summary[name] = {
            "expert_absent": float(frame[f"{name}_absent"].mean()),
            "low_probability": float(frame[f"{name}_low"].mean()),
            "mean_counted_diagnoses": float(frame[f"{name}_n"].mean()),
        }
    if "keyword" in present and "no_retrieval" in present:
        summary["keyword_minus_no_retrieval"] = bootstrap(
            frame["keyword_absent"].to_numpy(), frame["no_retrieval_absent"].to_numpy()
        )
    return summary


def sentence_agreement(notes: dict) -> dict:
    compared = 0
    same = 0
    label_only = 0
    text_only = 0
    examples = []
    for record in notes.values():
        for key in ("no_rag", "with_keyword", "with_tfidf"):
            claims = as_claims(record.get(key))
            if not claims:
                continue
            for claim in claims:
                if claim.get("type") != "diagnosis":
                    continue
                text = str(claim.get("text") or "")
                left = set(affirmed_labels(claim))
                right = set(text_only_labels(text))
                compared += 1
                if left == right:
                    same += 1
                    continue
                if left and not right:
                    label_only += 1
                elif right and not left:
                    text_only += 1
                if len(examples) < 8:
                    examples.append({
                        "text": text[:240],
                        "label_aware": sorted(left),
                        "text_only": sorted(right),
                    })
    return {
        "diagnosis_claims": compared,
        "agreement": same / compared if compared else None,
        "json_label_only": label_only,
        "text_extractor_only": text_only,
        "both_nonempty_and_different": compared - same - label_only - text_only,
        "examples": examples,
        "limitation": (
            "The text extractor is another rule, not a rater and not the PTB-XL code. "
            "Agreement is not clinical validation."
        ),
    }


def borrow_no_rag(notes: dict, reference: dict) -> tuple[dict, int]:
    borrowed = 0
    out = {}
    for ecg_id, record in notes.items():
        record = dict(record)
        if "no_rag" not in record and ecg_id in reference and "no_rag" in reference[ecg_id]:
            record["no_rag"] = reference[ecg_id]["no_rag"]
            borrowed += 1
        out[ecg_id] = record
    return out, borrowed


def uncertainty(notes: dict, by_id, cards: list[dict]) -> dict:
    checks = {card["id"]: card["check"] for card in cards if card.get("check")}
    unig = unig_table()
    applications = {"no_check": 0, "supported": 0, "refuted": 0, "indeterminate": 0, "outside": 0}
    linked = {name: {"n": 0, "code_present": 0} for name in ("supported", "refuted", "indeterminate")}
    abstain_patients = 0
    refuted_patients = 0
    n = 0
    scp_of = scp_table(set(notes))
    for ecg_id, record in notes.items():
        claims = as_claims(record.get("with_keyword"))
        if claims is None or ecg_id not in by_id:
            continue
        facts = facts_for(by_id[ecg_id])
        facts_unig = measurements_for(unig.loc[ecg_id], facts["sex"]) if ecg_id in unig.index else None
        patient_abstain = False
        patient_refuted = False
        n += 1
        for claim in claims:
            if claim.get("type") != "criterion" or not bool(claim.get("applies")):
                continue
            check = checks.get(claim.get("card_id"))
            if not check or check not in SCALAR:
                applications["no_check" if not check else "outside"] += 1
                continue
            status = verify(check, facts, facts_unig, MARGINS)
            applications[status] += 1
            if status == "indeterminate":
                patient_abstain = True
            elif status == "refuted":
                patient_refuted = True
            codes = SCP_FOR_CHECK.get(check)
            if codes and status in linked:
                linked[status]["n"] += 1
                if codes & scp_of.get(ecg_id, set()):
                    linked[status]["code_present"] += 1
        abstain_patients += int(patient_abstain)
        refuted_patients += int(patient_refuted)
    return {
        "n": n,
        "applications": applications,
        "patients_with_an_indeterminate_numeric_card": abstain_patients / n,
        "patients_with_a_refuted_numeric_card": refuted_patients / n,
        "linked_cardiologist_code_by_status": {
            name: {
                "n": row["n"],
                "code_present": None if row["n"] == 0 else row["code_present"] / row["n"],
            }
            for name, row in linked.items()
        },
        "note": (
            "Support already requires 12SL and Uni-G to agree and the 12SL value to sit outside the margin, "
            "so Uni-G cannot reject a supported card. The cardiologist statement code is not an input to that decision."
        ),
    }


def load_any(path: str) -> dict:
    return load_jsonl(path)


def main() -> None:
    ordinary = check_ordinary()
    if ordinary["failures"]:
        raise SystemExit(json.dumps(ordinary["failures"], indent=2))
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    cards = json.load(open(os.path.join(ROOT, "knowledge", "ecg_criteria_v2.json"), encoding="utf-8"))
    jobs = {
        "gpt_heldout": os.path.join(OUT, "heldout_generations.jsonl"),
        "gpt_numeric_cards": os.path.join(OUT, "clean_generations.jsonl"),
        "gpt_matched_prompt": os.path.join(OUT, "fair_generations.jsonl"),
        "gpt_constrained": os.path.join(OUT, "constrained_generations.jsonl"),
        "qwen3_8b": os.path.join(OUT, "open_models", "Qwen__Qwen3-8B.jsonl"),
        "qwen3_14b": os.path.join(OUT, "open_models", "Qwen__Qwen3-14B.jsonl"),
        "gemma3_12b": os.path.join(OUT, "open_models", "google__gemma-3-12b-it.jsonl"),
        "medgemma_4b": os.path.join(OUT, "open_models", "google__medgemma-4b-it.jsonl"),
        "deepseek_r1_8b": os.path.join(OUT, "open_models", "deepseek-ai__DeepSeek-R1-Distill-Llama-8B.jsonl"),
        "llama_3_1_8b": os.path.join(OUT, "open_models", "meta-llama__Llama-3.1-8B-Instruct.jsonl"),
    }
    heldout_notes = load_any(jobs["gpt_heldout"])
    borrow_from = {
        "gpt_numeric_cards": heldout_notes,
        "gpt_constrained": heldout_notes,
    }
    glm_heldout_path = os.path.join(OUT, "glm", "heldout_generations.jsonl")
    jobs.update({
        "glm_heldout": glm_heldout_path,
        "glm_numeric_cards": os.path.join(OUT, "glm", "clean_generations.jsonl"),
        "glm_matched_prompt": os.path.join(OUT, "glm", "fair_generations.jsonl"),
        "glm_constrained": os.path.join(OUT, "glm", "constrained_generations.jsonl"),
    })
    if os.path.exists(glm_heldout_path):
        glm_heldout = load_any(glm_heldout_path)
        borrow_from["glm_numeric_cards"] = glm_heldout
        borrow_from["glm_constrained"] = glm_heldout
    diagnosis = {}
    for name, path in jobs.items():
        if not os.path.exists(path):
            diagnosis[name] = {"missing": path}
            continue
        notes = load_any(path)
        if name in borrow_from:
            notes, borrowed = borrow_no_rag(notes, borrow_from[name])
        else:
            borrowed = 0
        summary = score_records(notes, by_id)
        if borrowed:
            summary["no_retrieval_source"] = "original no-card notes for the same patients, joined by ecg id"
            summary["borrowed_no_rag"] = borrowed
        diagnosis[name] = summary
    report = {
        "role": "exploratory rescore, not the prespecified cue-list analysis",
        "ordinary_sentences": ordinary,
        "real_sentence_agreement": sentence_agreement(heldout_notes),
        "diagnosis": diagnosis,
        "uncertainty_on_gpt_keyword_notes": uncertainty(heldout_notes, by_id, cards),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    brief = {
        "ordinary": {"matches": ordinary["matches"], "n": ordinary["n"], "bare": ordinary["bare_code_with_label_counts"]},
        "gpt": diagnosis["gpt_heldout"],
        "qwen": {key: diagnosis[key].get("keyword", {}).get("expert_absent") for key in ("qwen3_8b", "qwen3_14b")},
        "numeric": diagnosis.get("gpt_numeric_cards", {}).get("keyword_minus_no_retrieval"),
        "constrained": diagnosis.get("gpt_constrained", {}).get("keyword_minus_no_retrieval"),
        "sentences": report["real_sentence_agreement"],
        "linked": report["uncertainty_on_gpt_keyword_notes"].get("linked_cardiologist_code_by_status"),
    }
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
