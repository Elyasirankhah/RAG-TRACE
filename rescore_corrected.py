import ast
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from claim_scoring import affirms, filter_note, score_note
from independent_measurements import measurements_for
from retrieval_harm import MEASUREMENT_COLUMNS, cards, facts_for, load_cases

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "corrected_policy.json")


def load_notes() -> dict[int, dict]:
    found = {}
    with open(os.path.join(OUT, "confirm_generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                found[int(item["ecg_id"])] = item
    return found


def bootstrap(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(2)
    draws = []
    n = len(a)
    for _ in range(10000):
        index = rng.integers(0, n, n)
        draws.append(float(a[index].mean() - b[index].mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    discordant = only_a + only_b
    return {
        "difference": float(a.mean() - b.mean()),
        "ci95": [float(low), float(high)],
        "n_a_only": only_a,
        "n_b_only": only_b,
        "exact_mcnemar_p": float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else None,
    }


def main() -> None:
    library = cards()
    checks = {card["id"]: card["check"] for card in library}
    cases = load_cases()
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    notes = load_notes()
    unig_columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    unig = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(unig_columns),
    ).set_index("ecg_id")
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "scp_codes"])
    scp_of = {}
    for ecg_id, raw in database.loc[database["ecg_id"].isin(notes), ["ecg_id", "scp_codes"]].itertuples(index=False):
        scp_of[int(ecg_id)] = {code for code, likelihood in ast.literal_eval(raw).items() if likelihood > 0}

    attempted = len(pd.read_csv(os.path.join(OUT, "confirm_sample.csv")))
    rows = []
    audit = []
    for ecg_id, record in notes.items():
        source = by_id[ecg_id]
        facts = facts_for(source)
        sex = facts["sex"]
        facts_unig = measurements_for(unig.loc[ecg_id], sex) if ecg_id in unig.index else None
        if facts_unig is not None:
            facts_unig["probabilities"] = facts["probabilities"]
        labels = {name: int(source[f"y_{name}"]) for name in ["NORM", "MI", "STTC", "CD", "HYP"]}
        scp = scp_of.get(ecg_id, set())
        policies = {
            "no_retrieval": record["no_rag"],
            "always_retrieve": record["with_rag"],
            "check_12sl": filter_note(record["with_rag"], facts, facts_unig, checks, "12sl"),
            "check_both_algorithms": filter_note(record["with_rag"], facts, facts_unig, checks, "agree"),
        }
        item = {"ecg_id": ecg_id}
        for name, claims in policies.items():
            scored = score_note(claims, facts, labels, scp, checks)
            item[f"{name}_probability"] = int(len(scored["probability_errors"]) > 0)
            item[f"{name}_statement"] = int(len(scored["statement_errors"]) > 0)
            item[f"{name}_missed"] = len(scored["missed_statements"])
        rows.append(item)
        for claim in record["with_rag"]:
            if claim.get("type") == "diagnosis" and (
                facts["probabilities"].get(claim.get("label")) is None
                or facts["probabilities"].get(claim.get("label")) < 0.5
            ):
                audit.append({
                    "label": claim.get("label"),
                    "affirms": affirms(claim.get("text", "")),
                    "text": claim.get("text", ""),
                })

    frame = pd.DataFrame(rows)
    names = ["no_retrieval", "always_retrieve", "check_12sl", "check_both_algorithms"]
    report = {
        "scored_notes": int(len(frame)),
        "attempted_notes": int(attempted),
        "generation_failed": int(attempted - len(frame)),
        "rule": "Only affirmed sentences count. The check edits the retrieved note and withdraws a criterion unless measurements support it.",
        "probability_endpoint_case_rate": {name: float(frame[f"{name}_probability"].mean()) for name in names},
        "cardiologist_statement_endpoint_case_rate": {name: float(frame[f"{name}_statement"].mean()) for name in names},
        "mean_cardiologist_findings_not_stated": {name: float(frame[f"{name}_missed"].mean()) for name in names},
        "statement_endpoint_always_minus_both_algorithms": bootstrap(
            frame["always_retrieve_statement"].to_numpy(), frame["check_both_algorithms_statement"].to_numpy()
        ),
        "statement_endpoint_always_minus_12sl_check": bootstrap(
            frame["always_retrieve_statement"].to_numpy(), frame["check_12sl_statement"].to_numpy()
        ),
        "diagnosis_claims_below_0_5": {
            "n": len(audit),
            "affirmed": int(sum(item["affirms"] for item in audit)),
            "refused_or_hedged_with_explicit_denial": int(sum(not item["affirms"] for item in audit)),
            "affirmed_examples": [item["text"] for item in audit if item["affirms"]][:12],
            "denied_examples": [item["text"] for item in audit if not item["affirms"]][:8],
        },
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
