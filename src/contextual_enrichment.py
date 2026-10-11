"""
contextual_enrichment.py — LLM-based preamble generation for contextual chunks.

Reads API keys from environment (via .env):
  - GEMINI_API_KEY  → Gemini Flash (primary)
  - GROQ_API_KEY    → Groq llama-3.3-70b / llama-3.1-8b (fallback)

All responses are cached to cache/preambles.json keyed by sha256(original_chunk_text)
to avoid re-billing on repeated runs.

Schema update:
  - "text": original unmodified chunk text (source of truth, returned to caller)
  - "contextual_text": preamble + original chunk text (used for vector embedding)
"""

import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import sha256, load_json, save_json, CACHE_DIR, CHUNKS_DIR, NOVELS_DIR

from dotenv import load_dotenv
load_dotenv()

PREAMBLE_CACHE_FILE = CACHE_DIR / "preambles.json"
PREAMBLE_MAX_TOKENS = 100  # hard cap passed to prompt


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(chunk_text: str, novel_title: str) -> str:
    return (
        f"You are reading the novel '{novel_title}'. "
        f"In 1–2 sentences (maximum {PREAMBLE_MAX_TOKENS} words), "
        f"describe what happens in the following passage and its narrative context. "
        f"Be specific: name characters, events, or settings mentioned.\n\n"
        f"Passage:\n{chunk_text}\n\n"
        f"Context summary:"
    )


# ── LLM backends ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
    last_err = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini call failed: {last_err}")


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    models_to_try = ["qwen/qwen3.8-27b"]
    last_err = None
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=PREAMBLE_MAX_TOKENS,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Groq call failed: {last_err}")


def _get_backend() -> str:
    """Return 'groq', 'gemini', or 'fallback' based on available keys."""
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "fallback"


def _generate_preamble(prompt: str, primary_backend: str, chunk: dict = None, novel_title: str = "") -> tuple[str, str]:
    """
    Generate preamble with retries across Groq and Gemini backends.
    Strictly generates genuine LLM preambles.
    """
    backends = ["groq", "gemini"] if os.environ.get("GROQ_API_KEY") else ["gemini", "groq"]

    for attempt in range(30):
        for b in backends:
            try:
                if b == "groq" and os.environ.get("GROQ_API_KEY"):
                    return _call_groq(prompt), "groq"
                elif b == "gemini" and os.environ.get("GEMINI_API_KEY"):
                    return _call_gemini(prompt), "gemini"
            except Exception as e:
                continue
        time.sleep(min(60, (1.5 ** attempt) + random.uniform(1.0, 3.0)))

    # If all backends fail after 30 retries, raise exception to avoid silently injecting dummy text
    raise RuntimeError(f"Failed to generate LLM preamble for {novel_title} after 30 retries.")


import time


# ── Cache management ──────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if PREAMBLE_CACHE_FILE.exists():
        return load_json(PREAMBLE_CACHE_FILE)
    return {}


def _save_cache(cache: dict) -> None:
    save_json(cache, PREAMBLE_CACHE_FILE)


# ── Public interface ──────────────────────────────────────────────────────────

def enrich_chunk(chunk: dict, novel_title: str = "Pride and Prejudice") -> dict:
    """
    Enrich a single chunk dict with a contextual preamble.
    Preserves original text in chunk['text'] and sets chunk['contextual_text'].
    """
    backend = _get_backend()
    cache = _load_cache()

    original_text = chunk["text"]
    key = sha256(original_text)

    if key in cache:
        cache_entry = cache[key]
        if isinstance(cache_entry, dict):
            preamble = cache_entry["preamble"]
            source = cache_entry.get("source", "cache")
        else:
            preamble = cache_entry
            source = "cache"
    else:
        prompt = _build_prompt(original_text, novel_title)
        preamble, source = _generate_preamble(prompt, backend, chunk=chunk, novel_title=novel_title)
        cache[key] = {"preamble": preamble, "source": source}
        _save_cache(cache)

    chunk["text"] = original_text
    chunk["contextual_text"] = f"{preamble}\n\n{original_text}"
    chunk["preamble"] = preamble
    chunk["preamble_source"] = source
    chunk["strategy"] = "contextual"
    return chunk


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_cache_lock = threading.Lock()


def enrich_chunks(chunks: list[dict], novel_title: str, max_workers: int = 4) -> list[dict]:
    """
    For each chunk, generate/fetch a contextual preamble using concurrent LLM workers,
    preserving chunk['text'] as original and populating chunk['contextual_text'].
    Increments cache to disk after every 10 new preambles to ensure progress is saved.
    """
    backend = _get_backend()
    print(f"  Using primary LLM backend: {backend}")

    cache = _load_cache()
    stats = {"cache": 0, "gemini": 0, "groq": 0, "rule_based": 0}
    new_count = 0

    def process_single(c: dict) -> tuple[dict, str, str, str]:
        nonlocal new_count
        orig_text = c["text"]
        key = sha256(orig_text)

        with _cache_lock:
            if key in cache:
                cache_entry = cache[key]
                if isinstance(cache_entry, dict):
                    return c, cache_entry["preamble"], cache_entry.get("source", "groq"), "cache"
                return c, cache_entry, "groq", "cache"

        prompt = _build_prompt(orig_text, novel_title)
        preamble, source = _generate_preamble(prompt, backend, chunk=c, novel_title=novel_title)

        with _cache_lock:
            cache[key] = {"preamble": preamble, "source": source}

        return c, preamble, source, source

    futures = []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for c in chunks:
            futures.append(executor.submit(process_single, c))

        for f in tqdm(as_completed(futures), total=len(chunks), desc=f"Enriching {novel_title}", unit="chunk"):
            c, preamble, source, stat_key = f.result()
            stats[stat_key] = stats.get(stat_key, 0) + 1

            c["contextual_text"] = f"{preamble}\n\n{c['text']}"
            c["preamble"] = preamble
            c["preamble_source"] = source
            c["strategy"] = "contextual"
            if "__scene__" in c["id"]:
                c["id"] = c["id"].replace("__scene__", "__contextual__")

            completed_count += 1
            if completed_count % 25 == 0:
                with _cache_lock:
                    _save_cache(cache)

    with _cache_lock:
        _save_cache(cache)
    print(f"  Enrichment sources for {novel_title}: {stats}")
    if stats.get("rule_based", 0) > 5:
        print(f"  [WARNING] {stats['rule_based']} chunks fell back to rule-based synthesis! Disclose in README.")
    return chunks


def process_all_novels() -> dict[str, list[dict]]:
    """
    Load scene-strategy chunks for all 3 novels, enrich them into contextual chunks,
    and save as data/chunks/{novel}_contextual.json.
    """
    from chunker import chunk, chunk_and_save

    novels = ["Pride and Prejudice", "Jane Eyre", "Little Women"]
    enriched_results = {}

    # First ensure non-LLM chunks exist for all 3 novels
    for novel in novels:
        txt_path = NOVELS_DIR / f"{novel}.txt"
        if not txt_path.exists():
            print(f"  Preparing corpus for {novel}...")
            from corpus_prep import prep_novel
            prep_novel(novel)

        for strat in ["fixed", "paragraph", "scene"]:
            out_file = CHUNKS_DIR / f"{novel.replace(' ', '_')}_{strat}.json"
            if not out_file.exists():
                text = txt_path.read_text(encoding="utf-8")
                print(f"  Generating missing {strat} chunks for {novel}...")
                chunk_and_save(text, strat, novel)

    # Now generate contextual chunks based on scene chunks
    for novel in novels:
        scene_file = CHUNKS_DIR / f"{novel.replace(' ', '_')}_scene.json"
        contextual_file = CHUNKS_DIR / f"{novel.replace(' ', '_')}_contextual.json"

        if not scene_file.exists():
            print(f"  ERROR: Scene chunk file missing: {scene_file}")
            continue

        scene_chunks = load_json(scene_file)
        print(f"\nProcessing contextual enrichment for {novel} ({len(scene_chunks)} scene chunks)...")
        
        import copy
        c_chunks = copy.deepcopy(scene_chunks)
        enriched = enrich_chunks(c_chunks, novel)
        save_json(enriched, contextual_file)
        enriched_results[novel] = enriched
        print(f"  Saved contextual chunks -> {contextual_file}")

    return enriched_results


if __name__ == "__main__":
    print("=== Contextual Enrichment Scaling across Corpus ===\n")
    results = process_all_novels()
    
    all_chunks = []
    for novel, chunks in results.items():
        all_chunks.extend(chunks)

    print(f"\nTotal contextual chunks processed across corpus: {len(all_chunks)}")
    print("==========================================================================")
    print("SPOT-CHECK: 5 Random Contextual Chunks")
    print("==========================================================================")

    sample = random.sample(all_chunks, min(5, len(all_chunks)))
    for i, c in enumerate(sample, 1):
        print(f"\n[{i}] Chunk ID: {c['id']} | Novel: {c['novel']} | Chapter: {c['chapter']}")
        print(f"Preamble: {c['preamble']}")
        print(f"Original Text snippet (first 100 chars): {c['text'][:100]}...")
        print(f"Contextual Text snippet (first 150 chars): {c['contextual_text'][:150]}...")
        print("-" * 70)
