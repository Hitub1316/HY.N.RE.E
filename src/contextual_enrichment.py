"""
contextual_enrichment.py — LLM-based preamble generation for contextual chunks.

Reads API keys from environment (via .env):
  - GEMINI_API_KEY  → Gemini Flash (primary)
  - GROQ_API_KEY    → Groq llama-3.1-8b-instant (fallback)

All responses are cached to cache/preambles.json keyed by sha256(chunk_text)
to avoid re-billing on repeated runs.

Usage:
    from src.contextual_enrichment import enrich_chunks

    # chunks is a list[Chunk] with strategy='contextual'
    enriched = enrich_chunks(chunks, novel_title="Pride and Prejudice")
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import sha256, load_json, save_json, CACHE_DIR

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; keys must be in env already

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
    model = genai.GenerativeModel("gemini-3.6-flash")
    response = model.generate_content(prompt)
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    return ""


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PREAMBLE_MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


def _get_backend() -> str:
    """Return 'gemini' or 'groq' based on available keys."""
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    raise EnvironmentError(
        "No LLM API key found. Set GEMINI_API_KEY or GROQ_API_KEY in .env.\n"
        "Get a free Gemini key at: https://aistudio.google.com/app/apikey"
    )


def _generate_preamble(prompt: str, backend: str) -> str:
    if backend == "gemini":
        return _call_gemini(prompt)
    return _call_groq(prompt)


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
    Enrich a single chunk dict with a contextual preamble (LLM call, cached).
    """
    backend = _get_backend()
    cache = _load_cache()

    original_text = chunk["text"]
    key = sha256(original_text)

    if key in cache:
        preamble = cache[key]
    else:
        prompt = _build_prompt(original_text, novel_title)
        preamble = _generate_preamble(prompt, backend)
        cache[key] = preamble
        _save_cache(cache)

    chunk["text"] = f"{preamble}\n\n{original_text}"
    chunk["preamble"] = preamble
    chunk["strategy"] = "contextual"
    return chunk


def enrich_chunks(chunks: list[dict], novel_title: str) -> list[dict]:
    """
    For each chunk, generate a contextual preamble (LLM call, cached),
    and prepend it to chunk['text'].

    Mutates chunks in place and returns them.
    Only processes chunks with strategy='contextual'.
    """
    backend = _get_backend()
    print(f"  Using LLM backend: {backend}")

    cache = _load_cache()
    updated = 0

    for chunk in tqdm(chunks, desc="Generating preambles", unit="chunk"):
        if chunk.get("strategy") != "contextual":
            continue

        original_text = chunk["text"]
        key = sha256(original_text)

        if key in cache:
            preamble = cache[key]
        else:
            prompt = _build_prompt(original_text, novel_title)
            preamble = _generate_preamble(prompt, backend)
            cache[key] = preamble
            updated += 1
            # Save incrementally — avoids losing progress on interrupt
            if updated % 10 == 0:
                _save_cache(cache)

        chunk["text"] = f"{preamble}\n\n{original_text}"
        chunk["preamble"] = preamble  # store separately for inspection

    _save_cache(cache)
    print(f"  Preambles: {updated} new calls, {len(chunks) - updated} cached.")
    return chunks


def enrich_all_novels(corpus: list[dict]) -> dict[str, list[dict]]:
    """
    Convenience wrapper: load contextual chunks for each novel, enrich, save.
    Expects chunk files at data/chunks/{novel}_contextual.json.
    Returns dict of {novel_name: enriched_chunks}.
    """
    from utils import CHUNKS_DIR, load_json, save_json

    result = {}
    for novel_data in corpus:
        title = novel_data["novel_name"]
        chunk_file = CHUNKS_DIR / f"{title.replace(' ', '_')}_contextual.json"
        if not chunk_file.exists():
            print(f"  WARNING: {chunk_file} not found — run chunker.py first.")
            continue
        chunks = load_json(chunk_file)
        print(f"\n  Enriching: {title} ({len(chunks)} contextual chunks)")
        enriched = enrich_chunks(chunks, title)
        save_json(enriched, chunk_file)
        result[title] = enriched

    return result
