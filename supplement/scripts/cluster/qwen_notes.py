import json
import os
import sys
\
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
\
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
\
from heldout_notes import LIBRARY_PATH, SAMPLE, fit_tfidf, keyword_retrieve, library, note_prompt, tfidf_retrieve
from retrieval_harm import facts_for, load_cases
\
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "ptbxl_baseline", "open_models")
MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-2-9b-it",
]
\
\
def selected_models() -> list[str]:
    chosen = os.environ.get("MODEL_ID", "").strip()
    if chosen:
        return [chosen]
    return MODELS
def output_path(model_id: str) -> str:
    slug = model_id.replace("/", "__")
    return os.path.join(OUT_DIR, slug + ".jsonl")
def generate(model, tokenizer, model_id: str, messages: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in model_id:
        kwargs["enable_thinking"] = False
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
def done_ids(path: str) -> set[int]:
    found = set()
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                found.add(int(json.loads(line)["ecg_id"]))
    return found
def run_model(model_id: str, sample: pd.DataFrame, cards: list[dict], vectorizer, matrix) -> None:
    path = output_path(model_id)
    finished = done_ids(path)
    print(f"model {model_id} done {len(finished)} of {len(sample)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for _, row in sample.iterrows():
        ecg_id = int(row["ecg_id"])
        if ecg_id in finished:
            continue
        facts = facts_for(row)
        keyword = keyword_retrieve(facts, cards)
        tfidf = tfidf_retrieve(facts, cards, vectorizer, matrix)
        item = {
            "ecg_id": ecg_id,
            "model": model_id,
            "library": LIBRARY_PATH,
            "keyword_cards": [card["id"] for card in keyword],
            "tfidf_cards": [card["id"] for card in tfidf],
            "no_rag": generate(model, tokenizer, model_id, note_prompt(facts, None)),
            "with_keyword": generate(model, tokenizer, model_id, note_prompt(facts, keyword)),
            "with_tfidf": generate(model, tokenizer, model_id, note_prompt(facts, tfidf)),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item) + "\n")
        print(model_id, ecg_id, flush=True)
    del model
    torch.cuda.empty_cache()
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cards = library()
    vectorizer, matrix = fit_tfidf(cards)
    cases = load_cases()
    chosen = set(pd.read_csv(SAMPLE)["ecg_id"].astype(int))
    sample = cases[cases["ecg_id"].isin(chosen)]
    for model_id in selected_models():
        run_model(model_id, sample, cards, vectorizer, matrix)
if __name__ == "__main__":
    main()
