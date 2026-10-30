import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
\
import numpy as np
import pandas as pd
from scipy.stats import binomtest
\
from compare_policies import guarded_claims, patient_errors
from retrieval_harm import cards, facts_for, generate_case, load_cases, load_env, log
\
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
SAMPLE = os.path.join(OUT, "confirm_sample.csv")
RAW = os.path.join(OUT, "confirm_generations.jsonl")
FAILS = os.path.join(OUT, "confirm_failures.jsonl")
REPORT = os.path.join(OUT, "confirm_results.json")
N_PATIENTS = 500
SEED = 1
WORKERS = 4
BOOTSTRAPS = 10_000
\
\
def used_patient_ids(cases: pd.DataFrame) -> set[int]:
    used = set(pd.read_csv(os.path.join(OUT, "annotation_sample.csv"))["patient_id"].astype(int))
    path = os.path.join(OUT, "rag_generations.jsonl")
    if not os.path.exists(path):
        return used
    seen = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.append(json.loads(line)["ecg_id"])
    used.update(cases.loc[cases["ecg_id"].isin(seen), "patient_id"].astype(int))
    return used
def select_sample(cases: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(SAMPLE):
        chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
        return cases[cases["ecg_id"].isin(chosen)].copy()
    blocked = used_patient_ids(cases)
    fresh = cases[(cases["split"] == "test") & ~cases["patient_id"].astype(int).isin(blocked)]
    rng = np.random.default_rng(SEED)
    patients = fresh["patient_id"].astype(int).drop_duplicates().to_numpy()
    if len(patients) < N_PATIENTS:
        raise RuntimeError(f"only {len(patients)} untouched test patients")
    picked_patients = set(rng.choice(patients, size=N_PATIENTS, replace=False).tolist())
    rows = []
    for patient_id, group in fresh.groupby(fresh["patient_id"].astype(int)):
        if patient_id not in picked_patients:
            continue
        rows.append(group.sample(n=1, random_state=int(rng.integers(0, 10_000))))
    sample = pd.concat(rows, ignore_index=True)
    sample[["ecg_id", "patient_id"]].to_csv(SAMPLE, index=False)
    return sample
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
def generate(sample: pd.DataFrame, library: list[dict]) -> None:
    done = load_jsonl(RAW)
    failed = load_jsonl(FAILS)
    pending = [
        row for _, row in sample.iterrows() if int(row["ecg_id"]) not in done and int(row["ecg_id"]) not in failed
    ]
    log(f"confirm sample {len(sample)} pending {len(pending)} model {os.environ.get('API_MODEL')}")
    lock = threading.Lock()
    \
    def finish(ecg_id: int, future) -> None:
        try:
            item = future.result()
        except Exception as exc:
            item = {"ecg_id": ecg_id, "error": f"{type(exc).__name__}: {exc}"}
            with lock, open(FAILS, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            log(f"FAIL ecg {ecg_id}: {item['error']}")
            return
        with lock, open(RAW, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
        log(f"done {ecg_id}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(generate_case, row, library): int(row["ecg_id"]) for row in pending}
        for future in as_completed(futures):
            finish(futures[future], future)
def bootstrap_difference(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    difference = float(a.mean() - b.mean())
    draws = []
    n = len(a)
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, n, n)
        draws.append(float(a[index].mean() - b[index].mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    discordant = b_only + c_only
    p_value = float(binomtest(b_only, discordant, 0.5).pvalue) if discordant else None
    return {
        "difference": difference,
        "ci95": [float(low), float(high)],
        "n_a_only": b_only,
        "n_b_only": c_only,
        "exact_mcnemar_p": p_value,
    }
def score(sample: pd.DataFrame) -> dict:
    library = {card["id"]: card for card in cards()}
    checks = {card_id: card["check"] for card_id, card in library.items()}
    by_id = {int(row["ecg_id"]): row for _, row in sample.iterrows()}
    done = load_jsonl(RAW)
    failed = load_jsonl(FAILS)
    rows = []
    for ecg_id, row in by_id.items():
        record = done.get(ecg_id)
        if record is None:
            rows.append({"ecg_id": ecg_id, "patient_id": int(row["patient_id"]), "status": "generation_failed"})
            continue
        facts = facts_for(row)
        policies = {
            "no_retrieval": patient_errors(record["no_rag"], facts, checks),
            "always_retrieve": patient_errors(record["with_rag"], facts, checks),
            "measurement_check": patient_errors(guarded_claims(record, library, facts), facts, checks),
        }
        item = {
            "ecg_id": ecg_id,
            "patient_id": int(row["patient_id"]),
            "status": "scored",
        }
        for name, result in policies.items():
            item[name] = int(len(result["unsupported"]) > 0)
            item[f"{name}_count"] = len(result["unsupported"])
        rows.append(item)
    frame = pd.DataFrame(rows)
    scored = frame[frame["status"] == "scored"]
    names = ("no_retrieval", "always_retrieve", "measurement_check")
    rates = {name: float(scored[name].mean()) for name in names}
    rng = np.random.default_rng(2)
    paired = {
        "always_retrieve_minus_measurement_check": bootstrap_difference(
            scored["always_retrieve"].to_numpy(), scored["measurement_check"].to_numpy(), rng
        ),
        "always_retrieve_minus_no_retrieval": bootstrap_difference(
            scored["always_retrieve"].to_numpy(), scored["no_retrieval"].to_numpy(), rng
        ),
        "measurement_check_minus_no_retrieval": bootstrap_difference(
            scored["measurement_check"].to_numpy(), scored["no_retrieval"].to_numpy(), rng
        ),
    }
    attempted = len(frame)
    n_failed = int((frame["status"] == "generation_failed").sum())
    report = {
        "n_patients": int(frame["patient_id"].nunique()),
        "attempted_ecgs": attempted,
        "scored_ecgs": int(len(scored)),
        "generation_failed_ecgs": n_failed,
        "failed_ecg_ids": [int(ecg_id) for ecg_id in frame.loc[frame["status"] == "generation_failed", "ecg_id"]],
        "primary_endpoint": "Case-level unsupported patient claim on scored ECGs.",
        "primary_any_unsupported_rate": rates,
        "primary_mean_unsupported_claims": {name: float(scored[f"{name}_count"].mean()) for name in names},
        "paired_patient_bootstrap": paired,
        "missing_note_bounds": {
            "failures_counted_as_clean": {
                name: float(scored[name].sum() / attempted) for name in names
            },
            "failures_counted_as_unsupported": {
                name: float((scored[name].sum() + n_failed) / attempted) for name in names
            },
        },
        "unresolved_failures_outside_files": int(attempted - len(scored) - len(failed)),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log(json.dumps(report, indent=2))
    return report
def main() -> None:
    load_env()
    if "azure" in os.environ.get("API_BASE_URL", "").lower():
        raise RuntimeError("Azure is not used for this run")
    if not os.environ.get("API_KEY") or not os.environ.get("API_BASE_URL"):
        raise RuntimeError("API_BASE_URL and API_KEY must be set in .env")
    cases = load_cases()
    sample = select_sample(cases)
    overlap = used_patient_ids(cases) & set(sample["patient_id"].astype(int))
    if overlap:
        raise RuntimeError(f"sample includes {len(overlap)} patients from the first experiment")
    if sample["patient_id"].astype(int).nunique() != len(sample):
        raise RuntimeError("sample has more than one ECG for a patient")
    generate(sample, cards())
    score(sample)
if __name__ == "__main__":
    main()
