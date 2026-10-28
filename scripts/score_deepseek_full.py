import ast
import json
import os

import pandas as pd

from claim_scoring import filter_note
from heldout_notes import check_map, library
from independent_measurements import measurements_for
from retrieval_harm import MEASUREMENT_COLUMNS, facts_for, load_cases
from revision_evals import evaluate
from revision_evals import library as full_library
from score_qwen import claims_of, score_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "ptbxl_baseline", "open_models", "deepseek-ai__DeepSeek-R1-Distill-Llama-8B.jsonl")
OUT = os.path.join(ROOT, "results", "ptbxl_baseline", "deepseek_full_results.json")
INDEPENDENT = os.path.join(ROOT, "results", "ptbxl_baseline", "deepseek_full_independent.json")


def policies(record, facts, facts_unig, checks):
    keyword = record["with_keyword"]
    return {
        "no_retrieval": record["no_rag"],
        "keyword": keyword,
        "tfidf": record["with_tfidf"],
        "check_12sl": filter_note(keyword, facts, facts_unig, checks, "12sl"),
        "check_both": filter_note(keyword, facts, facts_unig, checks, "agree"),
    }


def main() -> None:
    cards = library()
    checks = check_map(cards)
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    unig_columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    unig = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(unig_columns),
    ).set_index("ecg_id")
    needed = set()
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                needed.add(int(json.loads(line)["ecg_id"]))
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "scp_codes"])
    scp_of = {}
    for ecg_id, raw in database.loc[database["ecg_id"].isin(needed), ["ecg_id", "scp_codes"]].itertuples(index=False):
        scp_of[int(ecg_id)] = {code for code, likelihood in ast.literal_eval(raw).items() if likelihood > 0}
    report = score_file("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", RAW, OUT, checks, by_id, unig, scp_of)
    print(json.dumps(report, indent=2))
    notes = {}
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            no_rag = claims_of(item.get("no_rag", ""))
            keyword = claims_of(item.get("with_keyword", ""))
            tfidf = claims_of(item.get("with_tfidf", ""))
            if no_rag is None or keyword is None or tfidf is None:
                continue
            notes[int(item["ecg_id"])] = {"no_rag": no_rag, "with_keyword": keyword, "with_tfidf": tfidf}
    full = {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", **evaluate(notes, cases, full_library(), policies)}
    with open(INDEPENDENT, "w", encoding="utf-8") as handle:
        json.dump(full, handle, indent=2)
    arms = full["arms"]
    print(json.dumps({
        "scored": full["scored_notes"],
        "no_release": arms["no_retrieval"]["criterion_rejected_by_12sl_or_unchecked"],
        "kw_release": arms["keyword"]["criterion_rejected_by_12sl_or_unchecked"],
        "tfidf_release": arms["tfidf"]["criterion_rejected_by_12sl_or_unchecked"],
        "after_release": arms["check_12sl"]["criterion_rejected_by_12sl_or_unchecked"],
        "no_expert": arms["no_retrieval"]["diagnosis_absent_from_expert_label"],
        "kw_expert": arms["keyword"]["diagnosis_absent_from_expert_label"],
        "kw_unig": arms["keyword"]["criterion_rejected_by_unig"],
        "after_unig": arms["check_12sl"]["criterion_rejected_by_unig"],
        "both_statement": arms["check_both"]["statement_mismatch"],
        "release_diff": full["contrasts"]["release_keyword_minus_no_retrieval"],
        "expert_diff": full["contrasts"]["expert_diagnosis_keyword_minus_no_retrieval"],
    }, indent=2))


if __name__ == "__main__":
    main()
