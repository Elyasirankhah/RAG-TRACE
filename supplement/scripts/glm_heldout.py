import json
import os
import time
import urllib.error
import urllib.request
\
import heldout_notes
from retrieval_harm import claim_list, load_env as read_dotenv, log, parse_json_content
\
ROOT = os.path.dirname(os.path.abspath(__file__))
GLM_OUT = os.path.join(ROOT, "results", "ptbxl_baseline", "glm")
BASE = "https://api.z.ai/api/paas/v4"
MODEL = "glm-5.3-flash"
\
\
def apply_glm() -> None:
    read_dotenv()
    key = os.environ.get("glm", "").strip()
    if not key:
        raise SystemExit("glm key is missing from .env")
    os.environ["API_BASE_URL"] = BASE
    os.environ["API_KEY"] = key
    os.environ["API_MODEL"] = MODEL
def complete(messages: list[dict]) -> dict:
    apply_glm()
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 6000,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
        }
    ).encode()
    last_error = None
    for attempt in range(1, 5):
        request = urllib.request.Request(
            f"{BASE}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode())
            choice = payload["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip():
                raise RuntimeError("empty note content")
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
            detail = detail.replace(os.environ.get("API_KEY", ""), "[redacted]")
            log(f"retry {attempt}: {type(exc).__name__} {detail}")
            time.sleep(2 * attempt)
    raise RuntimeError(last_error)
def redirect() -> None:
    os.makedirs(GLM_OUT, exist_ok=True)
    heldout_notes.complete = complete
    heldout_notes.load_env = apply_glm
    heldout_notes.PROTOCOL = os.path.join(GLM_OUT, "heldout_protocol.json")
    heldout_notes.RAW = os.path.join(GLM_OUT, "heldout_generations.jsonl")
    heldout_notes.FAILS = os.path.join(GLM_OUT, "heldout_failures.jsonl")
    heldout_notes.REPORT = os.path.join(GLM_OUT, "heldout_results.json")
    heldout_notes.PILOT_RAW = os.path.join(GLM_OUT, "heldout_pilot.jsonl")
def smoke() -> None:
    redirect()
    apply_glm()
    cards = heldout_notes.library()
    cases = heldout_notes.load_cases()
    sample = heldout_notes.freeze_sample(cases).head(1)
    vectorizer, matrix = heldout_notes.fit_tfidf(cards)
    row = sample.iloc[0]
    item = heldout_notes.generate_row(row, cards, vectorizer, matrix)
    claims = [len(item["no_rag"]), len(item["with_keyword"]), len(item["with_tfidf"])]
    if item["finish_reasons"] != ["stop", "stop", "stop"] or min(claims) < 1:
        raise SystemExit(f"smoke failed {item['finish_reasons']} claims {claims}")
    path = os.path.join(GLM_OUT, "smoke.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in item.items() if key != "raw"}, handle)
    log(json.dumps({"smoke": "passed", "ecg_id": item["ecg_id"], "claims": claims, "model": item["models"]}, indent=2))
def main() -> None:
    redirect()
    smoke()
    heldout_notes.prepare(heldout_notes.load_cases(), heldout_notes.library(), *heldout_notes.fit_tfidf(heldout_notes.library()))
    cards = heldout_notes.library()
    cases = heldout_notes.load_cases()
    vectorizer, matrix = heldout_notes.fit_tfidf(cards)
    heldout_notes.pilot(cases, cards, vectorizer, matrix)
    sample = heldout_notes.freeze_sample(cases)
    heldout_notes.run_generation(sample, heldout_notes.RAW, heldout_notes.FAILS, cards, vectorizer, matrix)
    heldout_notes.score(cases, cards)
if __name__ == "__main__":
    main()
