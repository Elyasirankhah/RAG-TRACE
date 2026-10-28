import ast
import json
import os
import sys

import pandas as pd

from claim_scoring import filter_note, score_note
from heldout_notes import check_map, library
from independent_measurements import measurements_for
from rescore_corrected import bootstrap
from retrieval_harm import MEASUREMENT_COLUMNS, facts_for, load_cases, parse_json_content

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
OPEN = os.path.join(OUT, "open_models")
MODELS = [
    ("Qwen/Qwen3-8B", "Qwen__Qwen3-8B.jsonl", "qwen_results.json"),
    ("Qwen/Qwen3-14B", "Qwen__Qwen3-14B.jsonl", "qwen14_results.json"),
    ("google/gemma-3-12b-it", "google__gemma-3-12b-it.jsonl", "gemma3_results.json"),
    ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama__Llama-3.1-8B-Instruct.jsonl", "llama_results.json"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai__DeepSeek-R1-Distill-Llama-8B.jsonl", "deepseek_results.json"),
]


def claims_of(raw: str) -> list[dict] | None:
    if not isinstance(raw, str):
        return None
    found = None
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = raw.find("{", start)
        if index < 0:
            break
        try:
            payload, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
            found = payload["claims"]
        start = index + 1
    if found is not None:
        return found
    try:
        payload = parse_json_content(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    claims = payload.get("claims", []) if isinstance(payload, dict) else None
    return claims if isinstance(claims, list) else None


def score_file(model_name: str, raw_path: str, report_path: str, checks, by_id, unig, scp_of) -> dict:
    notes = {}
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                notes[int(item["ecg_id"])] = item
    parsed = {"no_rag": 0, "with_keyword": 0, "with_tfidf": 0}
    for record in notes.values():
        for key in parsed:
            if claims_of(record.get(key, "")) is not None:
                parsed[key] += 1
    rows = []
    for ecg_id, record in notes.items():
        source = by_id[ecg_id]
        facts = facts_for(source)
        facts_unig = measurements_for(unig.loc[ecg_id], facts["sex"]) if ecg_id in unig.index else None
        if facts_unig is not None:
            facts_unig["probabilities"] = facts["probabilities"]
        labels = {name: int(source[f"y_{name}"]) for name in ["NORM", "MI", "STTC", "CD", "HYP"]}
        scp = scp_of.get(ecg_id, set())
        keyword = claims_of(record["with_keyword"])
        no_rag = claims_of(record["no_rag"])
        tfidf = claims_of(record["with_tfidf"])
        if keyword is None or no_rag is None or tfidf is None:
            continue
        policies = {
            "no_retrieval": no_rag,
            "keyword": keyword,
            "tfidf": tfidf,
            "check_12sl": filter_note(keyword, facts, facts_unig, checks, "12sl"),
            "check_both": filter_note(keyword, facts, facts_unig, checks, "agree"),
        }
        item = {"ecg_id": ecg_id}
        for name, claims in policies.items():
            scored = score_note(claims, facts, labels, scp, checks)
            item[f"{name}_probability"] = int(len(scored["probability_errors"]) > 0)
            item[f"{name}_statement"] = int(len(scored["statement_errors"]) > 0)
            item[f"{name}_missed"] = len(scored["missed_statements"])
        rows.append(item)
    frame = pd.DataFrame(rows)
    names = ["no_retrieval", "keyword", "tfidf", "check_12sl", "check_both"]
    if frame.empty:
        blank = {name: None for name in names}
        probability = statement = missed = blank
        contrast = None
    else:
        probability = {name: float(frame[f"{name}_probability"].mean()) for name in names}
        statement = {name: float(frame[f"{name}_statement"].mean()) for name in names}
        missed = {name: float(frame[f"{name}_missed"].mean()) for name in names}
        contrast = bootstrap(frame["keyword_probability"].to_numpy(), frame["check_both_probability"].to_numpy())
    report = {
        "model": model_name,
        "notes_in_file": int(len(notes)),
        "parsed_notes": parsed,
        "scored_notes": int(len(frame)),
        "unparsed_notes": int(len(notes) - len(frame)),
        "probability_endpoint_case_rate": probability,
        "statement_endpoint_case_rate": statement,
        "mean_findings_not_stated": missed,
        "probability_keyword_minus_both_algorithms": contrast,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    wanted = sys.argv[1:] or ["llama", "deepseek"]
    chosen = []
    for model_name, raw_name, report_name in MODELS:
        if any(token in model_name.lower() or token in raw_name.lower() for token in wanted):
            path = os.path.join(OPEN, raw_name)
            if os.path.exists(path):
                chosen.append((model_name, path, os.path.join(OUT, report_name)))
            else:
                print(f"missing {path}")
    if not chosen:
        return
    cards = library()
    checks = check_map(cards)
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    unig_columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    unig = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(unig_columns),
    ).set_index("ecg_id")
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "scp_codes"])
    needed = set()
    for _, raw_path, _ in chosen:
        with open(raw_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    needed.add(int(json.loads(line)["ecg_id"]))
    scp_of = {}
    for ecg_id, raw in database.loc[database["ecg_id"].isin(needed), ["ecg_id", "scp_codes"]].itertuples(index=False):
        scp_of[int(ecg_id)] = {code for code, likelihood in ast.literal_eval(raw).items() if likelihood > 0}
    for model_name, raw_path, report_path in chosen:
        report = score_file(model_name, raw_path, report_path, checks, by_id, unig, scp_of)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
