import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
\
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
\
from claim_scoring import affirms
from rescore_corrected import bootstrap
from retrieval_harm import chat, claim_list, load_env, log, tokens
\
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ds004504")
OUT = os.path.join(ROOT, "results", "eeg_check")
LIBRARY_PATH = os.path.join(ROOT, "knowledge", "eeg_criteria.json")
FEATURES = os.path.join(OUT, "features.csv")
SAMPLE = os.path.join(OUT, "sample.csv")
PROTOCOL = os.path.join(OUT, "protocol.json")
RAW = os.path.join(OUT, "generations.jsonl")
FAILS = os.path.join(OUT, "failures.jsonl")
REPORT = os.path.join(OUT, "results.json")
PILOT_RAW = os.path.join(OUT, "pilot.jsonl")
GROUPS = ["A", "F", "C"]
POSTERIOR = ["O1", "O2", "Pz"]
K = 3
WORKERS = 4
STOP = {"the", "and", "for", "with", "this", "that", "from", "eeg"}
QUERY_WORDS = {
    "A": "alzheimer slowing theta alpha posterior",
    "F": "frontotemporal dementia slowing rhythm",
    "C": "normal posterior alpha rhythm",
}
FEATURE_COLUMNS = ["peak_hz", "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "theta_alpha"]
\
\
def library() -> list[dict]:
    with open(LIBRARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)
def participants() -> pd.DataFrame:
    frame = pd.read_csv(os.path.join(DATA, "participants.tsv"), sep="\t")
    frame["subject"] = frame["participant_id"].str.replace("sub-", "", regex=False)
    return frame
def download() -> None:
    body = json.dumps(
        {
            "query": """
            query($datasetId: ID!, $tag: String!) {
              snapshot(datasetId: $datasetId, tag: $tag) {
                files(recursive: true) { filename size directory urls }
              }
            }
            """,
            "variables": {"datasetId": "ds004504", "tag": "1.0.9"},
        }
    ).encode()
    request = urllib.request.Request(
        "https://openneuro.org/crn/graphql",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    files = [
        item
        for item in payload["data"]["snapshot"]["files"]
        if not item.get("directory") and item["filename"].endswith(("_eeg.set", "_eeg.fdt"))
    ]
    log(f"eeg files {len(files)} bytes {sum(item['size'] for item in files)}")
    for index, item in enumerate(files, start=1):
        dest = os.path.join(DATA, item["filename"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == item["size"]:
            continue
        temporary = dest + ".part"
        request = urllib.request.Request(item["urls"][0], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response, open(temporary, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(temporary, dest)
        if index % 10 == 0 or index == len(files):
            log(f"downloaded {index} of {len(files)}")
def read_recording(subject: str) -> tuple[pd.DataFrame, float]:
    set_path = os.path.join(DATA, f"sub-{subject}", "eeg", f"sub-{subject}_task-eyesclosed_eeg.set")
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    names = [str(item.labels) for item in np.atleast_1d(mat["chanlocs"])]
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError(f"{subject} has no embedded EEG array")
    return pd.DataFrame(data.T, columns=names), float(mat["srate"])
def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))
def spectral_features(subject: str) -> dict:
    frame, srate = read_recording(subject)
    present = [name for name in POSTERIOR if name in frame.columns]
    if len(present) < 2:
        raise RuntimeError(f"{subject} missing posterior channels")
    start = int(2 * srate)
    stop = int(frame.shape[0] - 2 * srate)
    segment = frame.iloc[start:stop][present].to_numpy()
    segment = segment - np.nanmean(segment, axis=0, keepdims=True)
    butter = signal.butter(4, 1.0 / (srate / 2), btype="highpass")
    segment = signal.filtfilt(*butter, segment, axis=0)
    freqs, psd = signal.welch(segment, fs=srate, nperseg=int(4 * srate), noverlap=int(2 * srate), axis=0)
    mean_psd = np.nanmean(psd, axis=1)
    outside = ((freqs >= 2) & (freqs < 5)) | ((freqs > 15) & (freqs <= 30))
    slope, intercept = np.polyfit(np.log(freqs[outside]), np.log(mean_psd[outside] + 1e-12), 1)
    floor = np.exp(intercept + slope * np.log(np.maximum(freqs, 1e-6)))
    peak_band = (freqs >= 6) & (freqs <= 14)
    peak_hz = float(freqs[peak_band][np.argmax((mean_psd / floor)[peak_band])])
    delta = band_power(freqs, mean_psd, 1, 4)
    theta = band_power(freqs, mean_psd, 4, 8)
    alpha = band_power(freqs, mean_psd, 8, 13)
    beta = band_power(freqs, mean_psd, 13, 30)
    total = delta + theta + alpha + beta
    total = total if total > 0 else 1.0
    ratio = theta / alpha if alpha > 0 else 99.0
    return {
        "peak_hz": round(peak_hz, 3),
        "rel_delta": round(delta / total, 4),
        "rel_theta": round(theta / total, 4),
        "rel_alpha": round(alpha / total, 4),
        "rel_beta": round(beta / total, 4),
        "theta_alpha": round(ratio, 4),
    }
def leave_one_out(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = frame["Group"].to_numpy()
    probabilities = np.zeros((len(frame), len(GROUPS)))
    for index in range(len(frame)):
        train = np.ones(len(frame), dtype=bool)
        train[index] = False
        scaler = StandardScaler().fit(matrix[train])
        model = LogisticRegression(max_iter=400, class_weight="balanced")
        model.fit(scaler.transform(matrix[train]), labels[train])
        ordered = list(model.classes_)
        predicted = model.predict_proba(scaler.transform(matrix[index : index + 1]))[0]
        for name, value in zip(ordered, predicted):
            probabilities[index, GROUPS.index(name)] = value
    for name, column in zip(GROUPS, probabilities.T):
        frame[f"p_{name}"] = column
    return frame
def build_features() -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    people = participants()
    for _, person in people.iterrows():
        subject = person["subject"]
        row = {"subject": subject, "Group": person["Group"], "Age": int(person["Age"]), "Gender": person["Gender"], "MMSE": person["MMSE"]}
        row.update(spectral_features(subject))
        rows.append(row)
        log(f"features {subject}")
    frame = leave_one_out(pd.DataFrame(rows))
    frame.to_csv(FEATURES, index=False)
    log(frame[FEATURE_COLUMNS].describe().round(3).to_string())
    log(
        "prevalence "
        + f"peak<8 {(frame.peak_hz < 8).mean():.2f} "
        + f"ratio>=1 {(frame.theta_alpha >= 1).mean():.2f} "
        + f"theta>=0.25 {(frame.rel_theta >= 0.25).mean():.2f}"
    )
    return frame
def load_features() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES)
    frame["subject"] = frame["subject"].astype(str).str.zfill(3)
    return frame
def eeg_predicate(check: str, measurements: dict) -> bool:
    peak = measurements.get("peak_hz")
    ratio = measurements.get("theta_alpha")
    theta = measurements.get("rel_theta")
    if check == "peak_lt_8":
        return peak is not None and peak < 8
    if check == "peak_ge_10":
        return peak is not None and peak >= 10
    if check == "theta_alpha_ge_1":
        return ratio is not None and ratio >= 1
    if check == "theta_alpha_lt_0_5":
        return ratio is not None and ratio < 0.5
    if check == "rel_theta_ge_0_25":
        return theta is not None and theta >= 0.25
    if check == "rel_theta_lt_0_15":
        return theta is not None and theta < 0.15
    raise KeyError(check)
def facts_for(row: pd.Series) -> dict:
    measurements = {name: round(float(row[name]), 4) for name in FEATURE_COLUMNS}
    probabilities = {name: round(float(row[f"p_{name}"]), 3) for name in GROUPS}
    return {
        "subject": row["subject"],
        "sex": "female" if row["Gender"] == "F" else "male",
        "age": int(row["Age"]),
        "probabilities": probabilities,
        "asserted_labels": [name for name, value in probabilities.items() if value >= 0.5],
        "measurements": measurements,
    }
def card_document(card: dict) -> str:
    return card["text"] + " " + " ".join(card["keywords"])
def query_text(facts: dict) -> str:
    return " ".join(QUERY_WORDS[label] for label in facts["asserted_labels"] if label in QUERY_WORDS)
def keyword_retrieve(facts: dict, cards: list[dict]) -> list[dict]:
    query = tokens(query_text(facts)) - STOP
    if not query:
        return []
    scored = []
    for card in cards:
        overlap = len(query & (tokens(card_document(card)) - STOP))
        if overlap:
            scored.append((overlap, card["id"], card))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[:K]
def tfidf_retrieve(facts: dict, cards: list[dict], vectorizer, matrix) -> list[dict]:
    query = query_text(facts)
    if not query.strip():
        return []
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    order = np.argsort(-scores)[:K]
    return [(float(scores[index]), cards[index]["id"], cards[index]) for index in order if scores[index] > 0]
def note_prompt(facts: dict, retrieved: list[tuple] | None) -> list[dict]:
    shared = (
        "Return one JSON object with a claims list of at most 8 claims. "
        "Each claim has type, text, and optional label, name, value, card_id, applies. "
        "Use only the supplied measurement numbers. Do not invent values. "
        "A diagnosis label must be A, F, or C. "
        "Affirm a diagnosis only when its probability is at least 0.5, and then the text must say that it is present. "
        "Include one limit claim stating that this is not a clinical diagnosis."
    )
    if retrieved is None:
        system = shared + " Allowed types: diagnosis, measurement, limit. Do not cite external criteria."
        payload = facts
    else:
        system = (
            shared
            + " Allowed types: diagnosis, measurement, criterion, limit. "
            + "A criterion claim must set card_id to one of the retrieved ids and applies to true or false."
        )
        payload = {
            **facts,
            "retrieved_cards": [{"id": card["id"], "text": card["text"]} for _, _, card in retrieved],
        }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}]
def check_map(cards: list[dict]) -> dict[str, str]:
    return {card["id"]: card["check"] for card in cards if card.get("check")}
def filter_note(claims: list[dict], measurements: dict, checks: dict[str, str]) -> list[dict]:
    kept = []
    for claim in claims:
        claim = dict(claim)
        if claim.get("type") == "criterion" and bool(claim.get("applies")):
            check = checks.get(claim.get("card_id"))
            allowed = bool(check) and eeg_predicate(check, measurements)
            if not allowed:
                claim["applies"] = False
                text = claim.get("text", "")
                claim["text"] = text.rstrip() + " Not applied to this subject: the measurements do not establish this criterion."
        kept.append(claim)
    return kept
def score_claims(claims: list[dict], facts: dict, group: str, checks: dict[str, str]) -> dict:
    probability_errors = []
    missed_group = 0
    affirmed_groups = set()
    for claim in claims:
        if claim.get("type") == "diagnosis":
            if not affirms(claim.get("text", "")):
                continue
            label = claim.get("label")
            if not isinstance(label, str):
                probability_errors.append("affirmed_diagnosis_below_0.5")
                continue
            affirmed_groups.add(label)
            probability = facts["probabilities"].get(label)
            if probability is None or probability < 0.5:
                probability_errors.append("affirmed_diagnosis_below_0.5")
        elif claim.get("type") == "criterion" and bool(claim.get("applies")):
            check = checks.get(claim.get("card_id"))
            if not check or not eeg_predicate(check, facts["measurements"]):
                probability_errors.append("criterion_not_supported")
    if group not in affirmed_groups:
        missed_group = 1
    return {"probability_error": int(len(probability_errors) > 0), "missed_group": missed_group}
def done_ids(path: str) -> set[str]:
    found = set()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                found.add(str(json.loads(line)["subject"]).zfill(3))
    return found
def generate_one(row: pd.Series, cards, vectorizer, matrix) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
    return {
        "subject": row["subject"],
        "group": row["Group"],
        "keyword_cards": [card["id"] for _, _, card in keyword],
        "tfidf_cards": [card["id"] for _, _, card in tfidf],
        "no_rag": claim_list(chat(note_prompt(facts, None))),
        "with_keyword": claim_list(chat(note_prompt(facts, keyword))),
        "with_tfidf": claim_list(chat(note_prompt(facts, tfidf))),
    }
def write_generation(path: str, frame: pd.DataFrame) -> None:
    load_env()
    os.makedirs(OUT, exist_ok=True)
    cards = library()
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([card_document(card) for card in cards])
    finished = done_ids(path)
    pending = [row for _, row in frame.iterrows() if row["subject"] not in finished]
    log(f"pending {len(pending)} of {len(frame)} -> {path}")
    lock = __import__("threading").Lock()
    \
    def save(item: dict) -> None:
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_one, row, cards, vectorizer, matrix): row["subject"] for row in pending}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                save(future.result())
                log(f"saved {subject}")
            except Exception as exc:
                with open(FAILS, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"subject": subject, "error": str(exc)}) + "\n")
                log(f"failed {subject}: {type(exc).__name__}")
def prepare() -> None:
    os.makedirs(OUT, exist_ok=True)
    frame = load_features()
    rng = np.random.default_rng(21)
    pilot = []
    for group in GROUPS:
        members = frame.loc[frame["Group"] == group, "subject"].to_numpy()
        pilot.extend(rng.choice(members, size=2, replace=False).tolist())
    test = frame.loc[~frame["subject"].isin(pilot), ["subject", "Group"]]
    test.to_csv(SAMPLE, index=False)
    digest = hashlib.sha256(open(LIBRARY_PATH, "rb").read()).hexdigest()
    protocol = {
        "dataset": "OpenNeuro ds004504 v1.0.9",
        "recordings": "eyes closed, 19 channels, 500 Hz",
        "measurements": "posterior O1 O2 Pz Welch spectrum",
        "criteria_sha256": digest,
        "numeric_checks": ["peak_lt_8", "peak_ge_10", "theta_alpha_ge_1", "theta_alpha_lt_0_5", "rel_theta_ge_0_25", "rel_theta_lt_0_15"],
        "probabilities": "leave-one-out logistic regression on spectral features",
        "pilot_subjects": pilot,
        "test_subjects": int(len(test)),
        "retrieval_k": K,
        "model": os.environ.get("API_MODEL", ""),
    }
    if not os.path.exists(PROTOCOL):
        with open(PROTOCOL, "w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2)
    log(f"pilot {pilot} test {len(test)}")
def pilot() -> None:
    load_env()
    frame = load_features()
    with open(PROTOCOL, encoding="utf-8") as handle:
        chosen = json.load(handle)["pilot_subjects"]
    write_generation(PILOT_RAW, frame[frame["subject"].isin(chosen)])
    bad = 0
    with open(PILOT_RAW, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            for key in ("no_rag", "with_keyword", "with_tfidf"):
                if not isinstance(item[key], list) or not item[key]:
                    bad += 1
    if bad:
        raise RuntimeError(f"pilot has {bad} empty notes")
    log("pilot passed")
def run() -> None:
    load_env()
    if not os.path.exists(PILOT_RAW):
        raise RuntimeError("pilot has not been run")
    frame = load_features()
    sample = pd.read_csv(SAMPLE, dtype={"subject": str})
    sample["subject"] = sample["subject"].str.zfill(3)
    write_generation(RAW, frame[frame["subject"].isin(sample["subject"])])
def score() -> None:
    frame = load_features().set_index("subject", drop=False)
    cards = library()
    checks = check_map(cards)
    notes = {}
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                notes[str(item["subject"]).zfill(3)] = item
    rows = []
    for subject, record in notes.items():
        source = frame.loc[subject]
        facts = facts_for(source)
        policies = {
            "no_retrieval": record["no_rag"],
            "keyword": record["with_keyword"],
            "tfidf": record["with_tfidf"],
            "check": filter_note(record["with_keyword"], facts["measurements"], checks),
        }
        item = {"subject": subject}
        for name, claims in policies.items():
            scored = score_claims(claims, facts, source["Group"], checks)
            item[f"{name}_probability"] = scored["probability_error"]
            item[f"{name}_missed"] = scored["missed_group"]
        rows.append(item)
    table = pd.DataFrame(rows)
    names = ["no_retrieval", "keyword", "tfidf", "check"]
    report = {
        "dataset": "ds004504",
        "scored_subjects": int(len(table)),
        "probability_endpoint_case_rate": {name: float(table[f"{name}_probability"].mean()) for name in names},
        "mean_true_group_not_affirmed": {name: float(table[f"{name}_missed"].mean()) for name in names},
        "keyword_minus_check": bootstrap(table["keyword_probability"].to_numpy(), table["check_probability"].to_numpy()),
        "criterion_prevalence": {
            "peak_lt_8": float((frame["peak_hz"] < 8).mean()),
            "theta_alpha_ge_1": float((frame["theta_alpha"] >= 1).mean()),
            "rel_theta_ge_0_25": float((frame["rel_theta"] >= 0.25).mean()),
        },
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
def main() -> None:
    import sys
    \
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    started = time.time()
    if command == "download":
        download()
    elif command == "features":
        build_features()
    elif command == "prepare":
        load_env()
        prepare()
    elif command == "pilot":
        pilot()
    elif command == "run":
        run()
    elif command == "score":
        score()
    else:
        raise SystemExit(f"unknown command {command}")
    log(f"{command} finished in {time.time() - started:.0f}s")
if __name__ == "__main__":
    main()
