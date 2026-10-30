import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
\
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
\
from claim_scoring import affirms
from rescore_corrected import bootstrap
from retrieval_harm import chat, claim_list, load_env, log, tokens
\
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "sleep-edf")
OUT = os.path.join(ROOT, "results", "sleep_check")
LIBRARY_PATH = os.path.join(ROOT, "knowledge", "sleep_criteria.json")
FEATURES = os.path.join(OUT, "epochs.csv")
SAMPLE = os.path.join(OUT, "sample.csv")
PROTOCOL = os.path.join(OUT, "protocol.json")
RAW = os.path.join(OUT, "generations.jsonl")
FAILS = os.path.join(OUT, "failures.jsonl")
REPORT = os.path.join(OUT, "results.json")
PILOT_RAW = os.path.join(OUT, "pilot.jsonl")
BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
EXPECTED = {
    "SC4001E0-PSG.edf": 48338048,
    "SC4001EC-Hypnogram.edf": 4620,
    "SC4002E0-PSG.edf": 51621248,
    "SC4002EC-Hypnogram.edf": 4580,
    "SC4011E0-PSG.edf": 51110528,
    "SC4011EH-Hypnogram.edf": 3896,
    "SC4012E0-PSG.edf": 51986048,
    "SC4012EC-Hypnogram.edf": 5082,
    "SC4021E0-PSG.edf": 51147008,
    "SC4021EH-Hypnogram.edf": 4804,
    "SC4022E0-PSG.edf": 50271488,
    "SC4022EJ-Hypnogram.edf": 5292,
    "SC4031E0-PSG.edf": 51438848,
    "SC4031EC-Hypnogram.edf": 3700,
    "SC4032E0-PSG.edf": 49833728,
    "SC4032EP-Hypnogram.edf": 3800,
}
NIGHTS = {
    "SC4001": "EC",
    "SC4002": "EC",
    "SC4011": "EH",
    "SC4012": "EC",
    "SC4021": "EH",
    "SC4022": "EJ",
    "SC4031": "EC",
    "SC4032": "EP",
}
TRAIN = ["SC4001", "SC4002", "SC4011", "SC4012"]
TEST = ["SC4021", "SC4022", "SC4031", "SC4032"]
STAGES = ["W", "N1", "N2", "N3", "R"]
FEATURE_COLUMNS = ["slow_frac", "spindle_sec", "rel_delta", "rel_theta", "rel_alpha", "rel_sigma"]
K = 3
WORKERS = 4
STOP = {"the", "and", "for", "with", "this", "that", "from", "eeg", "sleep"}
QUERY_WORDS = {
    "W": "wake alpha rhythm",
    "N1": "n1 theta vertex",
    "N2": "n2 spindle sigma",
    "N3": "n3 slow wave delta amplitude",
    "R": "rem rapid eye",
}
STAGE_MAP = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage n1": "N1",
    "sleep stage 2": "N2",
    "sleep stage n2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "sleep stage n3": "N3",
    "sleep stage r": "R",
    "sleep stage rem": "R",
}
\
\
def download() -> None:
    os.makedirs(DATA, exist_ok=True)
    for night, suffix in NIGHTS.items():
        for name in (f"{night}E0-PSG.edf", f"{night}{suffix}-Hypnogram.edf"):
            dest = os.path.join(DATA, name)
            expected = EXPECTED.get(name)
            if os.path.exists(dest) and (expected is None or os.path.getsize(dest) == expected):
                log(f"have {name}")
                continue
            log(f"get {name}")
            temporary = dest + ".part"
            command = ["curl.exe", "-L", "--retry", "3", "--retry-delay", "2", "-o", temporary]
            if os.path.exists(temporary) and os.path.getsize(temporary) > 0:
                command.extend(["-C", "-"])
            command.append(BASE + name)
            subprocess.run(command, check=True)
            os.replace(temporary, dest)
def _header_field(handle, count: int, width: int) -> list[str]:
    block = handle.read(count * width)
    return [block[i : i + width].decode("latin1", "ignore").strip() for i in range(0, len(block), width)]
def read_edf(path: str):
    """Read a classic EDF or EDF+ file. Signal headers are stored field by field."""
    with open(path, "rb") as handle:
        header = handle.read(256)
        n_signals = int(header[252:256])
        n_records = int(header[236:244])
        record_sec = float(header[244:252])
        header_bytes = int(header[184:192])
        labels = _header_field(handle, n_signals, 16)
        _header_field(handle, n_signals, 80)
        units = _header_field(handle, n_signals, 8)
        pmin = [float(item) for item in _header_field(handle, n_signals, 8)]
        pmax = [float(item) for item in _header_field(handle, n_signals, 8)]
        dmin = [float(item) for item in _header_field(handle, n_signals, 8)]
        dmax = [float(item) for item in _header_field(handle, n_signals, 8)]
        _header_field(handle, n_signals, 80)
        samples = [int(float(item)) for item in _header_field(handle, n_signals, 8)]
        handle.read(n_signals * 32)
        consumed = 256 + n_signals * 256
        if header_bytes > consumed:
            handle.read(header_bytes - consumed)
        signals = []
        for index, label in enumerate(labels):
            signals.append(
                {
                    "label": label,
                    "unit": units[index],
                    "pmin": pmin[index],
                    "pmax": pmax[index],
                    "dmin": dmin[index],
                    "dmax": dmax[index],
                    "n": samples[index],
                }
            )
        values = {item["label"]: [] for item in signals if item["label"] != "EDF Annotations"}
        notes = []
        for _ in range(n_records):
            for item in signals:
                raw = handle.read(item["n"] * 2)
                if len(raw) < item["n"] * 2:
                    break
                if item["label"] == "EDF Annotations":
                    notes.append(raw)
                    continue
                ints = np.frombuffer(raw, dtype="<i2").astype(np.float64)
                span = item["dmax"] - item["dmin"]
                scale = 0.0 if span == 0 else (item["pmax"] - item["pmin"]) / span
                values[item["label"]].append((ints - item["dmin"]) * scale + item["pmin"])
    series = {name: np.concatenate(parts) for name, parts in values.items() if parts}
    rates = {item["label"]: item["n"] / record_sec for item in signals if item["label"] != "EDF Annotations"}
    units_map = {item["label"]: item["unit"] for item in signals}
    return series, rates, units_map, parse_annotations(notes)
def parse_annotations(records: list[bytes]) -> list[tuple[float, float, str]]:
    found = []
    for raw in records:
        text = raw.decode("latin1", "ignore").replace("\x00", "")
        for piece in text.split("\x14"):
            pass
        cursor = 0
        blob = raw.decode("latin1", "ignore")
        while cursor < len(blob):
            if blob[cursor] != "+":
                cursor += 1
                continue
            end = blob.find("\x14", cursor)
            if end < 0:
                break
            head = blob[cursor:end]
            cursor = end + 1
            label_end = blob.find("\x14", cursor)
            if label_end < 0:
                break
            label = blob[cursor:label_end].strip()
            cursor = label_end + 1
            if "\x15" in head:
                onset_text, duration_text = head.split("\x15", 1)
                duration = float(duration_text) if duration_text else 0.0
            else:
                onset_text, duration = head, 0.0
            try:
                onset = float(onset_text)
            except ValueError:
                continue
            if label:
                found.append((onset, duration, label))
    return found
def channel_of(series: dict) -> str:
    for name in series:
        lowered = name.lower()
        if "fpz" in lowered or "fpz-cz" in lowered:
            return name
    for name in series:
        if name.lower().startswith("eeg"):
            return name
    raise RuntimeError("no EEG channel")
def stage_epochs(annotations: list[tuple[float, float, str]]) -> list[tuple[float, str]]:
    epochs = []
    for onset, duration, label in annotations:
        stage = STAGE_MAP.get(label.strip().lower())
        if stage is None or duration <= 0:
            continue
        start = onset
        while start + 30 <= onset + duration + 1e-6:
            epochs.append((start, stage))
            start += 30
    return epochs
def slow_fraction(epoch: np.ndarray, srate: float) -> float:
    """Share of 1-second bins whose 0.5–2 Hz peak-to-peak amplitude is at least 75 µV."""
    window = int(srate)
    if window < 4 or len(epoch) < window:
        return 0.0
    hits = total = 0
    for start in range(0, len(epoch) - window + 1, window):
        total += 1
        if float(np.ptp(epoch[start : start + window])) >= 75.0:
            hits += 1
    return hits / total if total else 0.0
def epoch_measures(slow_share: float, spindle_env: np.ndarray, psd_bands: dict, wide: float, alpha_share: float, srate: float) -> dict:
    above = spindle_env >= 5.0
    longest = run = 0
    for flag in above:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    total = wide or 1.0
    return {
        "slow_frac": round(slow_share, 4),
        "spindle_sec": round(longest / srate, 3),
        "rel_delta": round(psd_bands["delta"] / total, 4),
        "rel_theta": round(psd_bands["theta"] / total, 4),
        "rel_alpha": round(alpha_share, 4),
        "rel_sigma": round(psd_bands["sigma"] / total, 4),
    }
def band_power(freqs, psd, low, high) -> float:
    mask = (freqs >= low) & (freqs < high)
    if int(mask.sum()) < 2:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))
def posterior_of(series: dict) -> str:
    for name in series:
        lowered = name.lower()
        if "pz-oz" in lowered or lowered.endswith("oz"):
            return name
    raise RuntimeError("no posterior EEG channel")
def night_epochs(night: str, suffix: str) -> list[dict]:
    series, rates, units, _ = read_edf(os.path.join(DATA, f"{night}E0-PSG.edf"))
    _, _, _, annotations = read_edf(os.path.join(DATA, f"{night}{suffix}-Hypnogram.edf"))
    frontal = channel_of(series)
    posterior = posterior_of(series)
    srate = float(rates[frontal])
    eeg = series[frontal]
    back = series[posterior]
    if units.get(frontal, "").lower() in {"mv", "millivolt"}:
        eeg = eeg * 1000.0
    if units.get(posterior, "").lower() in {"mv", "millivolt"}:
        back = back * 1000.0
    slow_filter = signal.butter(2, [0.5 / (srate / 2), 2.0 / (srate / 2)], btype="band")
    spindle_filter = signal.butter(2, [11.0 / (srate / 2), 16.0 / (srate / 2)], btype="band")
    slow = signal.filtfilt(*slow_filter, eeg)
    spindle = signal.filtfilt(*spindle_filter, eeg)
    envelope = np.abs(signal.hilbert(spindle))
    rows = []
    for onset, stage in stage_epochs(annotations):
        a = int(round(onset * srate))
        b = a + int(round(30 * srate))
        if a < 0 or b > len(eeg) or b > len(back):
            continue
        freqs, psd = signal.welch(eeg[a:b] - np.mean(eeg[a:b]), fs=srate, nperseg=min(len(eeg[a:b]), int(4 * srate)))
        wide = band_power(freqs, psd, 1, 30) or 1.0
        bands = {
            "delta": band_power(freqs, psd, 1, 4),
            "theta": band_power(freqs, psd, 4, 8),
            "sigma": band_power(freqs, psd, 11, 16),
        }
        back_freqs, back_psd = signal.welch(back[a:b] - np.mean(back[a:b]), fs=srate, nperseg=min(len(back[a:b]), int(4 * srate)))
        back_wide = band_power(back_freqs, back_psd, 1, 30) or 1.0
        alpha_share = band_power(back_freqs, back_psd, 8, 13) / back_wide
        row = {"night": night, "onset": round(onset, 1), "stage": stage, "split": "train" if night in TRAIN else "test"}
        row.update(epoch_measures(slow_fraction(slow[a:b], srate), envelope[a:b], bands, wide, alpha_share, srate))
        rows.append(row)
    log(f"{night} epochs {len(rows)}")
    return rows
def add_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame["split"].eq("train")
    scaler = StandardScaler().fit(frame.loc[train, FEATURE_COLUMNS])
    model = LogisticRegression(max_iter=400, class_weight="balanced")
    model.fit(scaler.transform(frame.loc[train, FEATURE_COLUMNS]), frame.loc[train, "stage"])
    for name in STAGES:
        frame[f"p_{name}"] = 0.0
    predicted = model.predict_proba(scaler.transform(frame.loc[~train, FEATURE_COLUMNS]))
    for column, name in enumerate(model.classes_):
        frame.loc[~train, f"p_{name}"] = predicted[:, column]
    return frame
def build_features() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for night, suffix in NIGHTS.items():
        rows.extend(night_epochs(night, suffix))
    frame = add_probabilities(pd.DataFrame(rows))
    frame.to_csv(FEATURES, index=False)
    test = frame[frame["split"].eq("test")]
    log(test["stage"].value_counts().to_string())
    log(
        "prevalence "
        + f"slow>=0.20 {(test.slow_frac >= 0.20).mean():.2f} "
        + f"slow<0.05 {(test.slow_frac < 0.05).mean():.2f} "
        + f"spindle>=0.5 {(test.spindle_sec >= 0.5).mean():.2f} "
        + f"spindle<0.3 {(test.spindle_sec < 0.3).mean():.2f} "
        + f"alpha>=0.15 {(test.rel_alpha >= 0.15).mean():.2f} "
        + f"alpha<0.08 {(test.rel_alpha < 0.08).mean():.2f}"
    )
    for stage, part in test.groupby("stage"):
        log(
            f"{stage} n={len(part)} "
            + f"slow={(part.slow_frac >= 0.20).mean():.2f} "
            + f"spindle={(part.spindle_sec >= 0.5).mean():.2f} "
            + f"alpha={(part.rel_alpha >= 0.15).mean():.2f}"
        )
def load_epochs() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES)
    frame["epoch_id"] = frame["night"] + "_" + frame["onset"].round(1).astype(str)
    return frame
def sleep_predicate(check: str, measurements: dict) -> bool:
    slow = measurements.get("slow_frac")
    spindle = measurements.get("spindle_sec")
    alpha = measurements.get("rel_alpha")
    if check == "slow_frac_ge_0_2":
        return slow is not None and slow >= 0.20
    if check == "slow_frac_lt_0_05":
        return slow is not None and slow < 0.05
    if check == "spindle_ge_0_5":
        return spindle is not None and spindle >= 0.5
    if check == "spindle_lt_0_3":
        return spindle is not None and spindle < 0.3
    if check == "alpha_ge_0_15":
        return alpha is not None and alpha >= 0.15
    if check == "alpha_lt_0_08":
        return alpha is not None and alpha < 0.08
    raise KeyError(check)
def facts_for(row: pd.Series) -> dict:
    measurements = {name: round(float(row[name]), 4) for name in FEATURE_COLUMNS}
    probabilities = {name: round(float(row[f"p_{name}"]), 3) for name in STAGES}
    return {
        "epoch_id": row["epoch_id"],
        "probabilities": probabilities,
        "asserted_labels": [name for name, value in probabilities.items() if value >= 0.5],
        "measurements": measurements,
    }
def library() -> list[dict]:
    with open(LIBRARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)
def card_document(card: dict) -> str:
    return card["text"] + " " + " ".join(card["keywords"])
def query_text(facts: dict) -> str:
    return " ".join(QUERY_WORDS[label] for label in facts["asserted_labels"] if label in QUERY_WORDS)
def keyword_retrieve(facts: dict, cards: list[dict]):
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
def tfidf_retrieve(facts: dict, cards: list[dict], vectorizer, matrix):
    query = query_text(facts)
    if not query.strip():
        return []
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    order = np.argsort(-scores)[:K]
    return [(float(scores[index]), cards[index]["id"], cards[index]) for index in order if scores[index] > 0]
def note_prompt(facts: dict, retrieved) -> list[dict]:
    shared = (
        "Return one JSON object with a claims list of at most 8 claims. "
        "Each claim has type, text, and optional label, name, value, card_id, applies. "
        "Use only the supplied measurement numbers. Do not invent values. "
        "A stage label must be W, N1, N2, N3, or R. "
        "Affirm a stage only when its probability is at least 0.5, and then the text must say that it is present. "
        "Include one limit claim stating that this is not a clinical sleep stage."
    )
    if retrieved is None:
        system = shared + " Allowed types: diagnosis, measurement, limit. Do not cite external criteria."
        payload = facts
    else:
        system = shared + " Allowed types: diagnosis, measurement, criterion, limit. A criterion claim must set card_id to a retrieved id and applies to true or false."
        payload = {**facts, "retrieved_cards": [{"id": card["id"], "text": card["text"]} for _, _, card in retrieved]}
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}]
def check_map(cards: list[dict]) -> dict:
    return {card["id"]: card["check"] for card in cards if card.get("check")}
def filter_note(claims: list[dict], measurements: dict, checks: dict) -> list[dict]:
    kept = []
    for claim in claims:
        claim = dict(claim)
        if claim.get("type") == "criterion" and bool(claim.get("applies")):
            check = checks.get(claim.get("card_id"))
            if not (check and sleep_predicate(check, measurements)):
                claim["applies"] = False
                claim["text"] = claim.get("text", "").rstrip() + " Not applied to this epoch: the measurements do not establish this criterion."
        kept.append(claim)
    return kept
def score_claims(claims: list[dict], facts: dict, stage: str, checks: dict) -> dict:
    errors = []
    affirmed = set()
    for claim in claims:
        if claim.get("type") == "diagnosis":
            if not affirms(claim.get("text", "")):
                continue
            label = claim.get("label")
            if not isinstance(label, str):
                errors.append("diagnosis")
                continue
            affirmed.add(label)
            if facts["probabilities"].get(label, 0) < 0.5:
                errors.append("diagnosis")
        elif claim.get("type") == "criterion" and bool(claim.get("applies")):
            check = checks.get(claim.get("card_id"))
            if not check or not sleep_predicate(check, facts["measurements"]):
                errors.append("criterion")
    return {"probability_error": int(bool(errors)), "missed_stage": int(stage not in affirmed)}
def done_ids(path: str) -> set[str]:
    found = set()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                found.add(json.loads(line)["epoch_id"])
    return found
def generate_one(row, cards, vectorizer, matrix) -> dict:
    facts = facts_for(row)
    keyword = keyword_retrieve(facts, cards)
    tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
    return {
        "epoch_id": row["epoch_id"],
        "stage": row["stage"],
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
    pending = [row for _, row in frame.iterrows() if row["epoch_id"] not in finished]
    log(f"pending {len(pending)} of {len(frame)}")
    lock = __import__("threading").Lock()
    \
    def save(item: dict) -> None:
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_one, row, cards, vectorizer, matrix): row["epoch_id"] for row in pending}
        for future in as_completed(futures):
            epoch_id = futures[future]
            try:
                save(future.result())
                log(f"saved {epoch_id}")
            except Exception as exc:
                with open(FAILS, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"epoch_id": epoch_id, "error": str(exc)}) + "\n")
                log(f"failed {epoch_id}: {type(exc).__name__}")
def prepare() -> None:
    os.makedirs(OUT, exist_ok=True)
    frame = load_epochs()
    test = frame[frame["split"].eq("test")].copy()
    rng = np.random.default_rng(7)
    chosen = []
    pilot = []
    for stage in STAGES:
        members = test.loc[test["stage"].eq(stage), "epoch_id"].to_numpy()
        if len(members) == 0:
            continue
        rng.shuffle(members)
        pilot.append(members[0])
        chosen.extend(members[1 : 1 + 12])
    sample = test[test["epoch_id"].isin(chosen)][["epoch_id", "night", "stage"]]
    sample.to_csv(SAMPLE, index=False)
    if not os.path.exists(PROTOCOL):
        protocol = {
            "dataset": "PhysioNet sleep-edfx 1.0.0 cassette",
            "nights_train": TRAIN,
            "nights_test": TEST,
            "channel": "Fpz-Cz for slow waves and spindles; Pz-Oz for alpha",
            "epoch_sec": 30,
            "rules": [
                "0.5-2 Hz peak-to-peak >= 75 uV for >= 20% of the epoch",
                "spindle envelope >= 5 uV for >= 0.5 s",
                "Pz-Oz relative alpha 8-13 Hz over 1-30 Hz >= 0.15",
            ],
            "criteria_sha256": hashlib.sha256(open(LIBRARY_PATH, "rb").read()).hexdigest(),
            "pilot_epochs": pilot,
            "test_epochs": int(len(sample)),
        }
        with open(PROTOCOL, "w", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2)
    log(f"pilot {len(pilot)} test {len(sample)}")
def pilot() -> None:
    load_env()
    frame = load_epochs()
    with open(PROTOCOL, encoding="utf-8") as handle:
        chosen = json.load(handle)["pilot_epochs"]
    write_generation(PILOT_RAW, frame[frame["epoch_id"].isin(chosen)])
    bad = 0
    with open(PILOT_RAW, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            for key in ("no_rag", "with_keyword", "with_tfidf"):
                if not item[key]:
                    bad += 1
    if bad:
        raise RuntimeError(f"pilot has {bad} empty notes")
    log("pilot passed")
def run() -> None:
    load_env()
    if not os.path.exists(PILOT_RAW):
        raise RuntimeError("pilot has not been run")
    frame = load_epochs()
    sample = pd.read_csv(SAMPLE)
    write_generation(RAW, frame[frame["epoch_id"].isin(sample["epoch_id"])])
def score() -> None:
    frame = load_epochs().set_index("epoch_id", drop=False)
    checks = check_map(library())
    notes = {}
    with open(RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                notes[item["epoch_id"]] = item
    rows = []
    for epoch_id, record in notes.items():
        source = frame.loc[epoch_id]
        facts = facts_for(source)
        policies = {
            "no_retrieval": record["no_rag"],
            "keyword": record["with_keyword"],
            "tfidf": record["with_tfidf"],
            "check": filter_note(record["with_keyword"], facts["measurements"], checks),
        }
        item = {"epoch_id": epoch_id}
        for name, claims in policies.items():
            scored = score_claims(claims, facts, source["stage"], checks)
            item[f"{name}_probability"] = scored["probability_error"]
            item[f"{name}_missed"] = scored["missed_stage"]
        rows.append(item)
    table = pd.DataFrame(rows)
    names = ["no_retrieval", "keyword", "tfidf", "check"]
    report = {
        "dataset": "sleep-edfx",
        "scored_epochs": int(len(table)),
        "probability_endpoint_case_rate": {name: float(table[f"{name}_probability"].mean()) for name in names},
        "mean_true_stage_not_affirmed": {name: float(table[f"{name}_missed"].mean()) for name in names},
        "keyword_minus_check": bootstrap(table["keyword_probability"].to_numpy(), table["check_probability"].to_numpy()),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
def main() -> None:
    import sys
    \
    command = sys.argv[1] if len(sys.argv) > 1 else "download"
    if command == "download":
        download()
    elif command == "features":
        build_features()
    elif command == "prepare":
        prepare()
    elif command == "pilot":
        pilot()
    elif command == "run":
        run()
    elif command == "score":
        score()
    else:
        raise SystemExit(f"unknown command {command}")
    log(f"{command} finished")
if __name__ == "__main__":
    main()
