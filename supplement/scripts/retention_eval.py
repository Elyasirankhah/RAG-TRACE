import json
import os
\
from claim_scoring import filter_note
from independent_measurements import holds, measurements_for
from retrieval_harm import MEASUREMENT_COLUMNS, facts_for, load_cases
from revision_evals import library
from score_qwen import claims_of
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "retention.json")
\
\
def unig_table():
    import pandas as pd
    \
    columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    return pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(columns),
    ).set_index("ecg_id")
def tally(notes: dict, cases, cards, keyword_of) -> dict:
    checks = {card["id"]: card["check"] for card in cards if card.get("check")}
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    unig = unig_table()
    bins = {
        "applied": 0,
        "unig_rejects": 0,
        "unig_rejects_withdrawn": 0,
        "unig_supports": 0,
        "unig_supports_kept": 0,
        "no_check": 0,
        "no_check_withdrawn": 0,
        "predicate_false": 0,
        "predicate_false_withdrawn": 0,
        "unig_missing": 0,
    }
    for ecg_id, record in notes.items():
        source = by_id[ecg_id]
        facts = facts_for(source)
        facts_unig = measurements_for(unig.loc[ecg_id], facts["sex"]) if ecg_id in unig.index else None
        keyword = keyword_of(record)
        checked = filter_note(keyword, facts, facts_unig, checks, "12sl")
        kept = {
            claim.get("card_id")
            for claim in checked
            if claim.get("type") == "criterion" and bool(claim.get("applies"))
        }
        for claim in keyword:
            if claim.get("type") != "criterion" or not bool(claim.get("applies")):
                continue
            bins["applied"] += 1
            check = checks.get(claim.get("card_id"))
            still = claim.get("card_id") in kept
            if not check:
                bins["no_check"] += 1
                bins["no_check_withdrawn"] += int(not still)
                continue
            from retrieval_harm import predicate
            \
            false_12 = not predicate(check, facts)
            if false_12:
                bins["predicate_false"] += 1
                bins["predicate_false_withdrawn"] += int(not still)
            verdict = None if facts_unig is None else holds(check, facts_unig)
            if verdict is True:
                bins["unig_supports"] += 1
                bins["unig_supports_kept"] += int(still)
            elif verdict is False:
                bins["unig_rejects"] += 1
                bins["unig_rejects_withdrawn"] += int(not still)
            else:
                bins["unig_missing"] += 1
    def frac(num, den):
        return None if den == 0 else num / den
    return {
        **bins,
        "fraction_unig_rejections_withdrawn": frac(bins["unig_rejects_withdrawn"], bins["unig_rejects"]),
        "fraction_unig_supported_kept": frac(bins["unig_supports_kept"], bins["unig_supports"]),
        "fraction_unchecked_withdrawn": frac(bins["no_check_withdrawn"], bins["no_check"]),
        "fraction_false_predicates_withdrawn": frac(bins["predicate_false_withdrawn"], bins["predicate_false"]),
    }
def load_closed() -> dict:
    notes = {}
    with open(os.path.join(OUT, "heldout_generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                notes[int(item["ecg_id"])] = item
    return notes
def load_qwen() -> dict:
    notes = {}
    path = os.path.join(OUT, "open_models", "Qwen__Qwen3-8B.jsonl")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            keyword = claims_of(item.get("with_keyword", ""))
            if keyword is None:
                continue
            notes[int(item["ecg_id"])] = {"with_keyword": keyword}
    return notes
def main() -> None:
    cards = library()
    cases = load_cases()
    report = {
        "rule": "12SL withdrawal. Uni-G is the judge and is not an input to this edit.",
        "gpt-4.1-mini": tally(load_closed(), cases, cards, lambda record: record["with_keyword"]),
        "Qwen3-8B": tally(load_qwen(), cases, cards, lambda record: record["with_keyword"]),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
