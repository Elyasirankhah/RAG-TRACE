import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
RAW = os.path.join(OUT, "rag_generations.jsonl")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TEXT = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction",
    "STTC": "ST-T change",
    "CD": "conduction disturbance",
    "HYP": "hypertrophy",
}
Q_COLUMNS = ["Q_Dur_II", "Q_Dur_III", "Q_Dur_aVF", "Q_Dur_V1", "Q_Dur_V2", "Q_Dur_V3", "Q_Dur_V4"]
MEASUREMENT_COLUMNS = [
    "HR_Ventr_Global",
    "PR_Int_Global",
    "QRS_Dur_Global",
    "QT_IntBazett_Global",
    "R_AxisFrontal_Global",
    "S_Amp_V1",
    "R_Amp_V5",
    "R_Amp_V6",
    "ST_Amp_II",
    *Q_COLUMNS,
]
WORKERS = 4


def load_env() -> None:
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def parse_json_content(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[-1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(os.path.join(OUT, "case_record.csv"))
    features = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "12sl_features.csv"),
        usecols=["ecg_id", *MEASUREMENT_COLUMNS],
    )
    merged = cases.merge(features, on="ecg_id", how="left", validate="one_to_one")
    merged["sex_label"] = np.where(merged["sex"].eq(0), "male", "female")
    return merged


def development_sample(cases: pd.DataFrame, count: int = 150) -> pd.DataFrame:
    pool = cases[cases["split"] == "calibrate"].copy()
    rng = np.random.default_rng(0)
    parts = []
    used = set()

    def take(frame: pd.DataFrame, n: int) -> None:
        remain = frame[~frame["ecg_id"].isin(used)]
        n = min(n, len(remain))
        if n == 0:
            return
        picked = remain.sample(n=n, random_state=int(rng.integers(0, 10_000)))
        used.update(picked["ecg_id"].tolist())
        parts.append(picked)

    take(pool[pool["hamming_error"] > 0].sort_values("confidence", ascending=False).head(400), 30)
    take(pool[pool["hamming_error"] == 0].sort_values("confidence", ascending=False).head(400), 30)
    take(pool.sort_values("confidence").head(400), 30)
    take(pool[pool["signal_quality_issue"] == 1], 30)
    take(pool[pool["y_HYP"] == 1], 30)
    take(pool, count - sum(len(part) for part in parts))
    return pd.concat(parts, ignore_index=True)


def cards() -> list[dict]:
    with open(os.path.join(ROOT, "knowledge", "ecg_criteria.json"), encoding="utf-8") as handle:
        return json.load(handle)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(facts: dict, library: list[dict], k: int = 3) -> list[dict]:
    # Measurement names such as Q_Dur and ST_Amp match the same three cards for
    # nearly every ECG. Retrieval uses the asserted diagnoses only.
    query = tokens(" ".join(facts["asserted_labels"]))
    if not query:
        return []
    scored = []
    for card in library:
        card_tokens = tokens(card["text"] + " " + " ".join(card["keywords"]) + " " + " ".join(card["labels"]))
        score = len(query & card_tokens) / math.sqrt(len(card_tokens))
        scored.append((score, card))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": round(score, 4), **card} for score, card in scored[:k]]


def facts_for(row: pd.Series) -> dict:
    measurements = {}
    for name in MEASUREMENT_COLUMNS:
        value = row[name]
        if pd.notna(value):
            measurements[name] = round(float(value), 3)
    asserted = [name for name in CLASSES if row[f"p_{name}"] >= 0.5]
    return {
        "ecg_id": int(row["ecg_id"]),
        "sex": row["sex_label"],
        "signal_quality_issue": bool(row["signal_quality_issue"]),
        "electrode_problem": bool(row["has_electrodes_problems"]),
        "probabilities": {name: round(float(row[f"p_{name}"]), 3) for name in CLASSES},
        "asserted_labels": asserted,
        "measurements": measurements,
    }


def predicate(check: str, facts: dict) -> bool:
    values = facts["measurements"]

    def read(name: str):
        return values.get(name)

    if check == "qrs_ge_120":
        value = read("QRS_Dur_Global")
        return value is not None and value >= 120
    if check == "pr_ge_200":
        value = read("PR_Int_Global")
        return value is not None and value >= 200
    if check == "axis_le_m30":
        value = read("R_AxisFrontal_Global")
        return value is not None and value <= -30
    if check == "sokolow_ge_3_5":
        s_v1, r_v5, r_v6 = read("S_Amp_V1"), read("R_Amp_V5"), read("R_Amp_V6")
        if s_v1 is None or r_v5 is None or r_v6 is None:
            return False
        return abs(s_v1) + max(r_v5, r_v6) >= 3.5
    if check == "qtc_long":
        value = read("QT_IntBazett_Global")
        if value is None:
            return False
        limit = 450 if facts["sex"] == "male" else 460
        return value >= limit
    if check == "st_ii_le_m0_1":
        value = read("ST_Amp_II")
        return value is not None and value <= -0.1
    if check == "q_ge_40":
        return any((read(name) or 0) >= 40 for name in Q_COLUMNS)
    if check == "hr_ge_100":
        value = read("HR_Ventr_Global")
        return value is not None and value >= 100
    if check == "qrs_lt_110":
        value = read("QRS_Dur_Global")
        return value is not None and value < 110
    if check == "pr_lt_120":
        value = read("PR_Int_Global")
        return value is not None and value < 120
    if check == "axis_ge_90":
        value = read("R_AxisFrontal_Global")
        return value is not None and value >= 90
    if check == "sokolow_lt_35":
        s_v1, r_v5, r_v6 = read("S_Amp_V1"), read("R_Amp_V5"), read("R_Amp_V6")
        if s_v1 is None or r_v5 is None or r_v6 is None:
            return False
        return abs(s_v1) + max(r_v5, r_v6) < 3.5
    if check == "hr_lt_60":
        value = read("HR_Ventr_Global")
        return value is not None and value < 60
    if check == "st_ii_ge_01":
        value = read("ST_Amp_II")
        return value is not None and value >= 0.1
    raise KeyError(check)


def prompt(facts: dict, retrieved: list[dict] | None) -> list[dict]:
    shared = (
        "Return one JSON object with a claims list. "
        "Each claim has type, text, and optional label, name, value, card_id, applies. "
        "Use only the supplied measurement numbers. Do not invent values. "
        "A diagnosis claim may use only a label whose probability is at least 0.5. "
        "Include one limit claim stating that this is not a clinical diagnosis."
    )
    if retrieved is None:
        system = shared + " Allowed types: diagnosis, measurement, limit. Do not cite external criteria."
        payload = facts
    else:
        system = (
            shared
            + " Allowed types: diagnosis, measurement, criterion, limit. "
            + "Use the retrieved cards to explain the ECG. "
            + "A criterion claim must set card_id to one of the retrieved ids and applies to true or false, "
            + "indicating whether that criterion holds for this patient."
        )
        payload = {**facts, "retrieved_cards": [{"id": card["id"], "text": card["text"]} for card in retrieved]}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload)},
    ]


def chat(messages: list[dict]) -> dict:
    base = os.environ["API_BASE_URL"].rstrip("/")
    url = f"{base}/chat/completions"
    body = json.dumps(
        {
            "model": os.environ["API_MODEL"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    last_error = None
    for attempt in range(1, 5):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            return parse_json_content(payload["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:300]
            log(f"retry {attempt}: {type(exc).__name__} {detail}")
            time.sleep(2 * attempt)
    raise RuntimeError(last_error)


def claim_list(payload: dict) -> list[dict]:
    claims = payload.get("claims", [])
    return claims if isinstance(claims, list) else []


def measurement_ok(claim: dict, facts: dict) -> bool:
    name = claim.get("name")
    if name not in facts["measurements"] or claim.get("value") is None:
        return False
    actual = facts["measurements"][name]
    try:
        reported = float(claim["value"])
    except (TypeError, ValueError):
        return False
    tolerance = 0.05 if "Amp" in name else 2
    return abs(reported - actual) <= tolerance


def score_arm(claims: list[dict], facts: dict, retrieved: list[dict] | None) -> dict:
    retrieved_ids = {card["id"] for card in retrieved} if retrieved else set()
    checks = {card["id"]: card["check"] for card in (retrieved or [])}
    unsupported_patient = []
    literature_ok = 0
    literature_bad = 0
    for claim in claims:
        kind = claim.get("type")
        if kind == "diagnosis":
            label = claim.get("label")
            if label not in facts["probabilities"] or facts["probabilities"][label] < 0.5:
                unsupported_patient.append({"reason": "diagnosis_without_probability", "claim": claim})
        elif kind == "measurement" and not measurement_ok(claim, facts):
            unsupported_patient.append({"reason": "measurement_mismatch", "claim": claim})
        elif kind == "criterion":
            card_id = claim.get("card_id")
            if card_id not in retrieved_ids:
                literature_bad += 1
                unsupported_patient.append({"reason": "unknown_criterion", "claim": claim})
                continue
            literature_ok += 1
            applies = bool(claim.get("applies"))
            holds = predicate(checks[card_id], facts)
            if applies != holds:
                unsupported_patient.append({"reason": "criterion_disagrees_with_measurements", "claim": claim})
    return {
        "n_claims": len(claims),
        "unsupported_patient": unsupported_patient,
        "literature_supported": literature_ok,
        "literature_unsupported": literature_bad,
    }


def predictor_row(row: pd.Series, retrieved: list[dict]) -> dict:
    return {
        "confidence": float(row["confidence"]),
        "signal_quality_issue": int(row["signal_quality_issue"]),
        "electrode_problem": int(row["has_electrodes_problems"]),
        "max_probability": float(max(row[f"p_{name}"] for name in CLASSES)),
        "n_asserted": int(sum(row[f"p_{name}"] >= 0.5 for name in CLASSES)),
        "mean_retrieval_score": float(np.mean([card["score"] for card in retrieved])),
        "human_validated": int(bool(row["validated_by_human"])),
        "initially_autogenerated": int(bool(row["initial_autogenerated_report"])),
    }


def generate_case(row: pd.Series, library: list[dict]) -> dict:
    facts = facts_for(row)
    retrieved = retrieve(facts, library)
    no_rag = claim_list(chat(prompt(facts, None)))
    with_rag = claim_list(chat(prompt(facts, retrieved)))
    no_score = score_arm(no_rag, facts, None)
    rag_score = score_arm(with_rag, facts, retrieved)
    no_reasons = {item["reason"] for item in no_score["unsupported_patient"]}
    rag_patient_errors = [
        item for item in rag_score["unsupported_patient"] if item["reason"] != "measurement_mismatch" or item["reason"] not in no_reasons
    ]
    clean_before = len(no_score["unsupported_patient"]) == 0
    harmed = clean_before and any(
        item["reason"] in {"diagnosis_without_probability", "criterion_disagrees_with_measurements", "unknown_criterion"}
        for item in rag_score["unsupported_patient"]
    )
    return {
        "ecg_id": int(row["ecg_id"]),
        "split": row["split"],
        "retrieved": [{"id": card["id"], "score": card["score"]} for card in retrieved],
        "no_rag": no_rag,
        "with_rag": with_rag,
        "no_rag_unsupported": len(no_score["unsupported_patient"]),
        "rag_unsupported": len(rag_score["unsupported_patient"]),
        "literature_supported": rag_score["literature_supported"],
        "literature_unsupported": rag_score["literature_unsupported"],
        "retrieval_induced_harm": bool(harmed),
        "features": predictor_row(row, retrieved),
    }


def load_done() -> dict[int, dict]:
    done = {}
    if not os.path.exists(RAW):
        return done
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                done[item["ecg_id"]] = item
    return done


def fit_predictor(records: list[dict]) -> dict:
    dev = [item for item in records if item["split"] == "calibrate"]
    test = [item for item in records if item["split"] == "test"]
    feature_names = list(dev[0]["features"])
    x_dev = np.array([[item["features"][name] for name in feature_names] for item in dev], dtype=float)
    y_dev = np.array([int(item["retrieval_induced_harm"]) for item in dev])
    x_test = np.array([[item["features"][name] for name in feature_names] for item in test], dtype=float)
    y_test = np.array([int(item["retrieval_induced_harm"]) for item in test])
    result = {
        "development_n": len(dev),
        "development_harm_rate": float(y_dev.mean()),
        "test_n": len(test),
        "test_harm_rate": float(y_test.mean()),
        "test_no_rag_unsupported_rate": float(np.mean([item["no_rag_unsupported"] > 0 for item in test])),
        "test_rag_unsupported_rate": float(np.mean([item["rag_unsupported"] > 0 for item in test])),
        "test_literature_claim_precision": None,
    }
    literature_total = sum(item["literature_supported"] + item["literature_unsupported"] for item in test)
    if literature_total:
        result["test_literature_claim_precision"] = float(
            sum(item["literature_supported"] for item in test) / literature_total
        )
    if len(set(y_dev)) < 2 or len(set(y_test)) < 2:
        result["predictor_auroc"] = None
        result["note"] = "Harm was absent on one split, so the predictor was not scored."
        return result
    scaler = StandardScaler()
    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(scaler.fit_transform(x_dev), y_dev)
    probability = model.predict_proba(scaler.transform(x_test))[:, 1]
    result["predictor_auroc"] = float(roc_auc_score(y_test, probability))
    result["predictor_auprc"] = float(average_precision_score(y_test, probability))
    result["coefficients"] = {
        name: round(float(weight), 3) for name, weight in zip(feature_names, model.coef_[0])
    }
    return result


def main() -> None:
    load_env()
    if not os.environ.get("API_KEY") or not os.environ.get("API_BASE_URL"):
        raise RuntimeError("API_BASE_URL and API_KEY must be set in .env")
    log(f"model {os.environ.get('API_MODEL')} at {os.environ['API_BASE_URL']}")
    os.makedirs(OUT, exist_ok=True)
    cases = load_cases()
    library = cards()
    test_ids = set(pd.read_csv(os.path.join(OUT, "annotation_sample.csv"))["ecg_id"].astype(int))
    selected = pd.concat(
        [development_sample(cases), cases[cases["ecg_id"].isin(test_ids)]],
        ignore_index=True,
    )
    selected = selected.drop_duplicates("ecg_id")
    done = load_done()
    pending = [row for _, row in selected.iterrows() if int(row["ecg_id"]) not in done]
    log(f"cases {len(selected)} pending {len(pending)}")
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_case, row, library): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            ecg_id = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                log(f"FAIL ecg {ecg_id}: {type(exc).__name__}: {exc}")
                continue
            with write_lock, open(RAW, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            done[ecg_id] = item
            log(f"done {len(done)}/{len(selected)} ecg {ecg_id} harm {item['retrieval_induced_harm']}")
    records = [done[int(ecg_id)] for ecg_id in selected["ecg_id"] if int(ecg_id) in done]
    summary = fit_predictor(records)
    with open(os.path.join(OUT, "retrieval_harm.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
