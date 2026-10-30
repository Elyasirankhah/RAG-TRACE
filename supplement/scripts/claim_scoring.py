import re
\
from independent_measurements import holds
from retrieval_harm import predicate
\
REFUSAL = re.compile(
    r"no strong evidence|no evidence|no clear evidence|no definitive|no pathologic|"
    r"not confirmed|not diagnosed|not asserted|not supported|not met|"
    r"does not reach|below the threshold|below threshold|"
    r"not sufficient|less probable|less likely|"
    r"no definitive diagnosis|cannot be made|"
    r"are not met|is not confirmed|is not diagnosed|is not asserted",
    re.I,
)
AFFIRM = re.compile(
    r"possible|probable|suggested|indicated|are present|is present|"
    r"shows a|show a|most likely|most probable|consistent with",
    re.I,
)
SCP_FOR_CHECK = {
    "qrs_ge_120": {"CLBBB", "CRBBB", "ILBBB", "IRBBB", "IVCD", "WPW"},
    "pr_ge_200": {"1AVB", "LPR"},
    "axis_le_m30": {"LAFB"},
    "sokolow_ge_3_5": {"LVH", "HVOLT"},
    "qtc_long": {"LNGQT"},
    "st_ii_le_m0_1": {"STD_"},
    "q_ge_40": {"QWAVE"},
    "hr_ge_100": {"STACH", "SVTAC"},
}
\
\
def affirms(text: str) -> bool:
    if not text or REFUSAL.search(text):
        return False
    return bool(AFFIRM.search(text))
def measurement_allows(check: str, facts_12: dict, facts_unig: dict | None, mode: str) -> bool:
    if mode == "12sl":
        return bool(predicate(check, facts_12))
    left = holds(check, facts_12)
    right = None if facts_unig is None else holds(check, facts_unig)
    return left is True and right is True
def filter_note(claims: list[dict], facts_12: dict, facts_unig: dict | None, checks: dict[str, str], mode: str) -> list[dict]:
    kept = []
    for claim in claims:
        claim = dict(claim)
        if claim.get("type") == "criterion" and bool(claim.get("applies")):
            check = checks.get(claim.get("card_id"))
            allowed = bool(check) and measurement_allows(check, facts_12, facts_unig, mode)
            if not allowed:
                claim["applies"] = False
                claim["text"] = (
                    claim.get("text", "").rstrip()
                    + " Not applied to this patient: the measurements do not establish this criterion."
                )
        kept.append(claim)
    return kept
def score_note(claims: list[dict], facts_12: dict, labels: dict[str, int], scp: set[str], checks: dict[str, str]) -> dict:
    probability_errors = []
    statement_errors = []
    applied = []
    missed_statements = []
    for claim in claims:
        kind = claim.get("type")
        if kind == "diagnosis":
            if not affirms(claim.get("text", "")):
                continue
            raw_label = claim.get("label")
            label_list = raw_label if isinstance(raw_label, list) else [raw_label]
            for label in label_list:
                if not isinstance(label, str):
                    probability_errors.append("affirmed_diagnosis_below_0.5")
                    statement_errors.append("affirmed_diagnosis_absent_from_labels")
                    continue
                probability = facts_12["probabilities"].get(label)
                if probability is None or probability < 0.5:
                    probability_errors.append("affirmed_diagnosis_below_0.5")
                if labels.get(label, 0) != 1:
                    statement_errors.append("affirmed_diagnosis_absent_from_labels")
        elif kind == "criterion" and bool(claim.get("applies")):
            card_id = claim.get("card_id")
            check = checks.get(card_id)
            if check is None:
                probability_errors.append("unknown_criterion")
                continue
            if check not in SCP_FOR_CHECK:
                if not predicate(check, facts_12):
                    probability_errors.append("criterion_not_supported_by_12sl")
                continue
            applied.append(check)
            if not predicate(check, facts_12):
                probability_errors.append("criterion_not_supported_by_12sl")
            if not (SCP_FOR_CHECK[check] & scp):
                statement_errors.append("criterion_not_in_cardiologist_statements")
    present_checks = {check for check, codes in SCP_FOR_CHECK.items() if codes & scp}
    for check in present_checks - set(applied):
        missed_statements.append(check)
    return {
        "probability_errors": probability_errors,
        "statement_errors": statement_errors,
        "missed_statements": missed_statements,
    }
