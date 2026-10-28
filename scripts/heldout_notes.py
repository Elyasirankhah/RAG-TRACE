import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from claim_scoring import filter_note, score_note
from independent_measurements import measurements_for
from rescore_corrected import bootstrap
from retrieval_harm import (
    CLASS_TEXT,
    MEASUREMENT_COLUMNS,
    claim_list,
    facts_for,
    load_cases,
    load_env,
    log,
    parse_json_content,
    tokens,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
LIBRARY_PATH = os.path.join(ROOT, "knowledge", "ecg_criteria_v2.json")
SAMPLE = os.path.join(OUT, "heldout_sample.csv")
PROTOCOL = os.path.join(OUT, "heldout_protocol.json")
RAW = os.path.join(OUT, "heldout_generations.jsonl")
FAILS = os.path.join(OUT, "heldout_failures.jsonl")
REPORT = os.path.join(OUT, "heldout_results.json")
PILOT_SAMPLE = os.path.join(OUT, "heldout_pilot_sample.csv")
PILOT_RAW = os.path.join(OUT, "heldout_pilot.jsonl")
SEED = 11
PILOT_SEED = 12
PILOT_N = 16
K = 3
WORKERS = 4
STOP = {"ecg", "the", "and", "for", "with", "this", "that", "from"}
QUERY_WORDS = {
    "NORM": "normal rhythm",
    "MI": "myocardial infarct infarction q wave",
    "STTC": "st depression elevation repolarization qt",
    "CD": "conduction qrs pr axis bundle",
    "HYP": "hypertrophy voltage sokolow",
}


def library() -> list[dict]:
    with open(LIBRARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def library_hash() -> str:
    data = open(LIBRARY_PATH, "rb").read()
    return hashlib.sha256(data).hexdigest()


def check_map(cards: list[dict]) -> dict[str, str]:
    return {card["id"]: card["check"] for card in cards if card.get("check")}


def query_text(facts: dict) -> str:
    return " ".join(QUERY_WORDS[label] for label in facts["asserted_labels"] if label in QUERY_WORDS)


def query_tokens(facts: dict) -> set[str]:
    return tokens(query_text(facts)) - STOP


def card_document(card: dict) -> str:
    # Superclass labels are not part of the document. Tagging every card with
    # NORM or MI made the shortest tagged card win for almost every ECG.
    return card["text"] + " " + " ".join(card["keywords"])


def keyword_retrieve(facts: dict, cards: list[dict], k: int = K) -> list[dict]:
    query = query_tokens(facts)
    if not query:
        return []
    scored = []
    for card in cards:
        card_tokens = tokens(card_document(card)) - STOP
        if not card_tokens:
            continue
        overlap = len(query & card_tokens)
        if overlap == 0:
            continue
        scored.append((overlap, card["id"], card))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"score": int(overlap), **card} for overlap, _, card in scored[:k]]


def fit_tfidf(cards: list[dict]):
    docs = [card_document(card) for card in cards]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def tfidf_retrieve(facts: dict, cards: list[dict], vectorizer, matrix, k: int = K) -> list[dict]:
    query = query_text(facts)
    if not query.strip():
        return []
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    order = [int(index) for index in np.argsort(-scores) if scores[index] > 0][:k]
    return [{"score": round(float(scores[index]), 4), **cards[index]} for index in order]


def note_prompt(facts: dict, retrieved: list[dict] | None) -> list[dict]:
    shared = (
        "Return one JSON object with a claims list of at most 8 claims. "
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


def complete(messages: list[dict]) -> dict:
    load_env()
    base = os.environ["API_BASE_URL"].rstrip("/")
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
            f"{base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            parsed = parse_json_content(content)
            return {
                "claims": claim_list(parsed),
                "model": payload.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"),
                "raw": content,
            }
        except Exception as exc:
            last_error = exc
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", "replace")[:300]
            log(f"retry {attempt}: {type(exc).__name__} {detail}")
            time.sleep(2 * attempt)
    raise RuntimeError(last_error)


def blocked_patients() -> set[int]:
    blocked = set()
    for name in ("annotation_sample.csv", "confirm_sample.csv", "noise_sample.csv"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            blocked.update(pd.read_csv(path)["patient_id"].astype(int))
    return blocked


def one_per_patient(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, group in frame.groupby(frame["patient_id"].astype(int)):
        rows.append(group.sample(n=1, random_state=int(rng.integers(0, 10_000))))
    return pd.concat(rows, ignore_index=True)


def freeze_sample(cases: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(SAMPLE):
        chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
        return cases[cases["ecg_id"].isin(chosen)].copy()
    fresh = cases[(cases["split"] == "test") & ~cases["patient_id"].astype(int).isin(blocked_patients())]
    sample = one_per_patient(fresh, SEED)
    sample[["ecg_id", "patient_id"]].to_csv(SAMPLE, index=False)
    return sample


def retrieval_diagnostic(cases: pd.DataFrame, cards: list[dict], vectorizer, matrix) -> dict:
    pool = one_per_patient(cases[cases["split"] == "calibrate"], 13).head(300)
    keyword_sets = []
    tfidf_sets = []
    keyword_distractor = tfidf_distractor = 0
    keyword_cards = tfidf_cards = 0
    for _, row in pool.iterrows():
        facts = facts_for(row)
        keyword = keyword_retrieve(facts, cards)
        tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
        keyword_sets.append(tuple(card["id"] for card in keyword))
        tfidf_sets.append(tuple(card["id"] for card in tfidf))
        for card in keyword:
            keyword_cards += 1
            keyword_distractor += int(not card.get("check"))
        for card in tfidf:
            tfidf_cards += 1
            tfidf_distractor += int(not card.get("check"))
    return {
        "calibration_patients": int(len(pool)),
        "keyword_unique_top3_sets": int(len(set(keyword_sets))),
        "tfidf_unique_top3_sets": int(len(set(tfidf_sets))),
        "keyword_unique_cards": int(len({card_id for item in keyword_sets for card_id in item})),
        "tfidf_unique_cards": int(len({card_id for item in tfidf_sets for card_id in item})),
        "keyword_distractor_fraction": keyword_distractor / keyword_cards if keyword_cards else None,
        "tfidf_distractor_fraction": tfidf_distractor / tfidf_cards if tfidf_cards else None,
    }


def write_protocol(sample: pd.DataFrame, diagnostic: dict) -> None:
    if os.path.exists(PROTOCOL):
        return
    load_env()
    protocol = {
        "frozen_before_test_generation": True,
        "clinician_review": "not part of this run",
        "test_patients": int(sample["patient_id"].nunique()),
        "test_ecgs": int(len(sample)),
        "one_ecg_per_patient": True,
        "seed": SEED,
        "excluded_patient_lists": ["annotation_sample.csv", "confirm_sample.csv", "noise_sample.csv"],
        "library": "knowledge/ecg_criteria_v2.json",
        "library_sha256": library_hash(),
        "requested_model": os.environ.get("API_MODEL"),
        "temperature": 0,
        "max_tokens": 2000,
        "primary_retrieval": "keyword overlap on asserted diagnosis words, k=3",
        "secondary_retrieval": "tfidf on asserted diagnosis words, k=3",
        "protection": "Withdraw an applied criterion from the primary retrieved note unless 12SL and Uni-G both support its check. Cards with no measurement check are withdrawn when applied.",
        "primary_endpoint": "Case rate of affirmed diagnoses below probability 0.5, or criteria applied when the 12SL predicate is false, or applied cards with no measurement check.",
        "primary_contrast": "Primary retrieved note minus the both-algorithm check.",
        "secondary_endpoints": [
            "cardiologist statement codes for the original eight checks only",
            "mean cardiologist findings left unstated",
            "same probability endpoint on the TF-IDF note",
        ],
        "calibration_retrieval_diagnostic": diagnostic,
        "open_weight_model": "Not run on the 4 GB laptop GPU. Cluster script: cluster/qwen_notes.sbatch",
        "pilot_passed": False,
    }
    with open(PROTOCOL, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)


def load_jsonl(path: str) -> dict[int, dict]:
    found = {}
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                found[int(item["ecg_id"])] = item
    return found


def generate_row(row: pd.Series, cards: list[dict], vectorizer, matrix) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
    no_rag = complete(note_prompt(facts, None))
    with_keyword = complete(note_prompt(facts, keyword))
    with_tfidf = complete(note_prompt(facts, tfidf))
    return {
        "ecg_id": int(row["ecg_id"]),
        "patient_id": int(row["patient_id"]),
        "keyword_cards": [card["id"] for card in keyword],
        "tfidf_cards": [card["id"] for card in tfidf],
        "no_rag": no_rag["claims"],
        "with_keyword": with_keyword["claims"],
        "with_tfidf": with_tfidf["claims"],
        "models": [no_rag["model"], with_keyword["model"], with_tfidf["model"]],
        "finish_reasons": [no_rag["finish_reason"], with_keyword["finish_reason"], with_tfidf["finish_reason"]],
        "usage": [no_rag["usage"], with_keyword["usage"], with_tfidf["usage"]],
        "raw": {"no_rag": no_rag["raw"], "with_keyword": with_keyword["raw"], "with_tfidf": with_tfidf["raw"]},
    }


def run_generation(sample: pd.DataFrame, raw_path: str, fail_path: str, cards: list[dict], vectorizer, matrix) -> None:
    done = load_jsonl(raw_path)
    failed = load_jsonl(fail_path) if fail_path else {}
    pending = [
        row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed
    ]
    log(f"sample {len(sample)} pending {len(pending)} model {os.environ.get('API_MODEL')}")
    lock = threading.Lock()

    def finish(ecg_id: int, future) -> None:
        try:
            item = future.result()
        except Exception as exc:
            item = {"ecg_id": ecg_id, "error": f"{type(exc).__name__}: {exc}"}
            if fail_path:
                with lock, open(fail_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
            log(f"FAIL ecg {ecg_id}: {item['error']}")
            return
        with lock, open(raw_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
        log(f"done {ecg_id}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(generate_row, row, cards, vectorizer, matrix): int(row["ecg_id"]) for row in pending
        }
        for future in as_completed(futures):
            finish(futures[future], future)


def prepare(cases: pd.DataFrame, cards: list[dict], vectorizer, matrix) -> None:
    sample = freeze_sample(cases)
    diagnostic = retrieval_diagnostic(cases, cards, vectorizer, matrix)
    write_protocol(sample, diagnostic)
    log(json.dumps({"sample": int(len(sample)), **diagnostic}, indent=2))


def pilot(cases: pd.DataFrame, cards: list[dict], vectorizer, matrix) -> None:
    if not os.path.exists(PILOT_SAMPLE):
        pool = cases[cases["split"] == "calibrate"]
        one_per_patient(pool, PILOT_SEED).head(PILOT_N)[["ecg_id", "patient_id"]].to_csv(PILOT_SAMPLE, index=False)
    chosen = set(pd.read_csv(PILOT_SAMPLE)["ecg_id"].astype(int))
    sample = cases[cases["ecg_id"].isin(chosen)].copy()
    run_generation(sample, PILOT_RAW, "", cards, vectorizer, matrix)
    notes = load_jsonl(PILOT_RAW)
    reasons = []
    models = []
    for note in notes.values():
        reasons.extend(note["finish_reasons"])
        models.extend(note["models"])
    passed = len(notes) == len(sample) and reasons.count("stop") == len(reasons) and len(reasons) > 0
    protocol = json.load(open(PROTOCOL, encoding="utf-8"))
    protocol["pilot_notes"] = int(len(notes))
    protocol["pilot_finish_reasons"] = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
    protocol["response_model"] = sorted(set(models))
    protocol["pilot_passed"] = bool(passed)
    with open(PROTOCOL, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    log(json.dumps({"pilot_passed": passed, "notes": len(notes), "models": protocol["response_model"]}, indent=2))
    if not passed:
        raise SystemExit(1)


def score(cases: pd.DataFrame, cards: list[dict]) -> None:
    import ast

    checks = check_map(cards)
    notes = load_jsonl(RAW)
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    unig_columns = ["ecg_id", *[name for name in MEASUREMENT_COLUMNS if name != "R_AxisFrontal_Global"], "QRS_AxisFront_Global"]
    unig = pd.read_csv(
        os.path.join(ROOT, "ptb-xl-plus", "features", "unig_features.csv"),
        usecols=lambda name: name in set(unig_columns),
    ).set_index("ecg_id")
    database = pd.read_csv(os.path.join(ROOT, "ptb-xl", "ptbxl_database.csv"), usecols=["ecg_id", "scp_codes"])
    scp_of = {}
    for ecg_id, raw in database.loc[database["ecg_id"].isin(notes), ["ecg_id", "scp_codes"]].itertuples(index=False):
        scp_of[int(ecg_id)] = {code for code, likelihood in ast.literal_eval(raw).items() if likelihood > 0}
    rows = []
    for ecg_id, record in notes.items():
        source = by_id[ecg_id]
        facts = facts_for(source)
        facts_unig = measurements_for(unig.loc[ecg_id], facts["sex"]) if ecg_id in unig.index else None
        if facts_unig is not None:
            facts_unig["probabilities"] = facts["probabilities"]
        labels = {name: int(source[f"y_{name}"]) for name in ["NORM", "MI", "STTC", "CD", "HYP"]}
        scp = scp_of.get(ecg_id, set())
        policies = {
            "no_retrieval": record["no_rag"],
            "keyword": record["with_keyword"],
            "tfidf": record["with_tfidf"],
            "check_12sl": filter_note(record["with_keyword"], facts, facts_unig, checks, "12sl"),
            "check_both": filter_note(record["with_keyword"], facts, facts_unig, checks, "agree"),
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
    attempted = int(len(pd.read_csv(SAMPLE)))
    report = {
        "scored_notes": int(len(frame)),
        "attempted_notes": attempted,
        "generation_failed": int(attempted - len(frame)),
        "probability_endpoint_case_rate": {name: float(frame[f"{name}_probability"].mean()) for name in names},
        "statement_endpoint_case_rate": {name: float(frame[f"{name}_statement"].mean()) for name in names},
        "mean_findings_not_stated": {name: float(frame[f"{name}_missed"].mean()) for name in names},
        "probability_keyword_minus_both_algorithms": bootstrap(
            frame["keyword_probability"].to_numpy(), frame["check_both_probability"].to_numpy()
        ),
        "probability_keyword_minus_12sl": bootstrap(
            frame["keyword_probability"].to_numpy(), frame["check_12sl_probability"].to_numpy()
        ),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "pilot", "run", "score"])
    args = parser.parse_args()
    load_env()
    cards = library()
    cases = load_cases()
    vectorizer, matrix = fit_tfidf(cards)
    if args.command == "prepare":
        prepare(cases, cards, vectorizer, matrix)
    elif args.command == "pilot":
        if not os.path.exists(PROTOCOL):
            prepare(cases, cards, vectorizer, matrix)
        pilot(cases, cards, vectorizer, matrix)
    elif args.command == "run":
        protocol = json.load(open(PROTOCOL, encoding="utf-8"))
        if not protocol.get("pilot_passed"):
            raise SystemExit("pilot has not passed")
        sample = freeze_sample(cases)
        run_generation(sample, RAW, FAILS, cards, vectorizer, matrix)
        score(cases, cards)
    else:
        score(cases, cards)


if __name__ == "__main__":
    main()
