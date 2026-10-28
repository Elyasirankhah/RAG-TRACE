import json
import math
import os
from collections import Counter

import numpy as np

from diagnosis_prf import gold_of, predicted
from exploratory_revision import as_claims
from heldout_notes import (
    card_document,
    keyword_retrieve,
    library,
    query_tokens,
    tfidf_retrieve,
    fit_tfidf,
    tokens,
    STOP,
)
from retrieval_harm import facts_for, load_cases
from revision_evals import load_jsonl

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "ptbxl_baseline")
REPORT = os.path.join(OUT, "offline_day.json")


def bm25_retrieve(facts, cards, k=3, k1=1.5, b=0.75):
    docs = [tokens(card_document(card)) - STOP for card in cards]
    query = query_tokens(facts)
    if not query:
        return []
    n_docs = len(docs)
    avgdl = sum(len(doc) for doc in docs) / n_docs
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    scores = np.zeros(n_docs)
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        length = len(doc) or 1
        total = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            n_term = df.get(term, 0)
            idf = math.log((n_docs - n_term + 0.5) / (n_term + 0.5) + 1.0)
            total += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * length / avgdl))
        scores[index] = total
    order = sorted(range(n_docs), key=lambda i: (-scores[i], cards[i]["id"]))
    return [cards[i] for i in order if scores[i] > 0][:k]


def retrieval_block(cases, cards):
    vectorizer, matrix = fit_tfidf(cards)
    checked = {card["id"] for card in cards if card.get("check")}
    rows = []
    for _, source in cases.iterrows():
        facts = facts_for(source)
        got = {
            "keyword": keyword_retrieve(facts, cards),
            "tfidf": tfidf_retrieve(facts, cards, vectorizer, matrix),
            "bm25": bm25_retrieve(facts, cards),
        }
        item = {
            "ecg_id": int(source["ecg_id"]),
            "asserted": list(facts["asserted_labels"]),
            "y_hyp": int(source["y_HYP"]),
            "p_hyp": float(source["p_HYP"]),
        }
        for name, picked in got.items():
            ids = [card["id"] for card in picked]
            item[name] = ids
            item[f"{name}_untestable"] = int(any(card["id"] not in checked for card in picked))
            item[f"{name}_sokolow"] = int("sokolow_lyon" in ids)
        rows.append(item)
    summary = {"n": len(rows), "library_cards": len(cards), "checked_cards": len(checked)}
    hyp_asserted = [row for row in rows if "HYP" in row["asserted"]]
    hyp_stored = [row for row in rows if row["y_hyp"] == 1]
    for name in ("keyword", "tfidf", "bm25"):
        summary[name] = {
            "patients_with_an_untestable_card": float(np.mean([row[f"{name}_untestable"] for row in rows])),
            "sokolow_when_hypertrophy_asserted": _rate(hyp_asserted, f"{name}_sokolow"),
            "sokolow_when_hypertrophy_stored": _rate(hyp_stored, f"{name}_sokolow"),
            "n_hypertrophy_asserted": len(hyp_asserted),
            "n_hypertrophy_stored": len(hyp_stored),
        }
    return summary


def _rate(rows, key):
    if not rows:
        return None
    return float(np.mean([row[key] for row in rows]))


def hypertrophy_audit(cases):
    by_id = {int(row["ecg_id"]): row for _, row in cases.iterrows()}
    notes = load_jsonl(os.path.join(OUT, "matched_constrained_generations.jsonl"))
    lost = []
    both = only_kw = neither = 0
    groups = {
        "p_at_least_0.5": {"n": 0, "no": 0, "key": 0},
        "p_below_0.5": {"n": 0, "no": 0, "key": 0},
    }
    for ecg_id, record in notes.items():
        row = by_id.get(int(ecg_id))
        if row is None or int(row["y_HYP"]) != 1:
            continue
        left = predicted(as_claims(record.get("no_rag")))
        right = predicted(as_claims(record.get("with_keyword")))
        no = "HYP" in left
        key = "HYP" in right
        bucket = groups["p_at_least_0.5" if float(row["p_HYP"]) >= 0.5 else "p_below_0.5"]
        bucket["n"] += 1
        bucket["no"] += int(no)
        bucket["key"] += int(key)
        if no and key:
            both += 1
        elif no and not key:
            only_kw += 0
            cards = record.get("keyword_cards") or []
            card_ids = [card.get("id") if isinstance(card, dict) else card for card in cards]
            texts = []
            for claim in as_claims(record.get("with_keyword")) or []:
                if isinstance(claim, dict):
                    texts.append(str(claim.get("text") or ""))
            blob = " ".join(texts).lower()
            lost.append({
                "ecg_id": int(ecg_id),
                "p_hyp": round(float(row["p_HYP"]), 3),
                "asserted": "HYP" in facts_for(row)["asserted_labels"],
                "retrieved_sokolow": "sokolow_lyon" in card_ids,
                "retrieved": card_ids,
                "keyword_mentions_hypertrophy": "hypertrophy" in blob,
                "keyword_labels": sorted(right),
            })
        elif key and not no:
            only_kw += 1
        else:
            neither += 1
    for bucket in groups.values():
        bucket["recall_no_cards"] = bucket["no"] / bucket["n"] if bucket["n"] else None
        bucket["recall_keyword"] = bucket["key"] / bucket["n"] if bucket["n"] else None
    return {
        "stored_hypertrophy": both + len(lost) + only_kw + neither,
        "kept_by_both": both,
        "lost_under_constraint": len(lost),
        "gained_under_constraint": only_kw,
        "missed_by_both": neither,
        "by_probability": groups,
        "lost": lost,
    }


def main():
    heldout = load_jsonl(os.path.join(OUT, "heldout_generations.jsonl"))
    cases = load_cases()
    cases = cases[cases["ecg_id"].astype(int).isin(set(heldout))].copy()
    cards = library()
    numeric = [card for card in cards if card.get("check")]
    report = {
        "no_api": True,
        "full_library": retrieval_block(cases, cards),
        "numeric_cards": retrieval_block(cases, numeric),
        "hypertrophy": hypertrophy_audit(cases),
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    hyp = report["hypertrophy"]
    print(json.dumps({
        "full_untestable": {name: report["full_library"][name]["patients_with_an_untestable_card"] for name in ("keyword", "tfidf", "bm25")},
        "numeric_sokolow_stored": {name: report["numeric_cards"][name]["sokolow_when_hypertrophy_stored"] for name in ("keyword", "tfidf", "bm25")},
        "lost": hyp["lost_under_constraint"],
        "by_probability": hyp["by_probability"],
        "lost_mention": sum(item["keyword_mentions_hypertrophy"] for item in hyp["lost"]),
        "lost_sokolow": sum(item["retrieved_sokolow"] for item in hyp["lost"]),
        "lost_asserted": sum(item["asserted"] for item in hyp["lost"]),
    }, indent=2))


if __name__ == "__main__":
    main()
