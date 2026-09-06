"""
utils.py — Shared helpers for HY.N.RE.E pipeline.
"""

import hashlib
import json
import os
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
NOVELS_DIR = ROOT / "novels"
DATA_DIR = ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EVAL_DIR = DATA_DIR / "eval"
CACHE_DIR = ROOT / "cache"
QDRANT_DIR = ROOT / "qdrant_storage"

for _d in (DATA_DIR, CHUNKS_DIR, EVAL_DIR, CACHE_DIR, QDRANT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Chunk schema ─────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str,
    novel: str,
    strategy: str,
    text: str,
    char_start: int,
    char_end: int,
    chapter: str = "",
) -> dict:
    """Return a Chunk dict conforming to the canonical schema."""
    return {
        "id": chunk_id,
        "novel": novel,
        "strategy": strategy,
        "text": text,
        "char_start": char_start,
        "char_end": char_end,
        "chapter": chapter,
    }


# ── Hashing ──────────────────────────────────────────────────────────────────

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict | list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Token counting (sentence-transformers tokenizer) ─────────────────────────

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def token_count(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))
