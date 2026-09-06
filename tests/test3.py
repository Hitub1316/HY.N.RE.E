"""
test4.py — Smoke test: contextual enrichment on a small batch (5-10 chunks).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contextual_enrichment import enrich_chunk  # adjust to actual function name

# Load one already-chunked file
with open("data/chunks/Pride_and_Prejudice_scene.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Take only first 8 chunks — not the whole novel
test_chunks = chunks[:8]

print(f"Testing contextual enrichment on {len(test_chunks)} chunks\n")

for c in test_chunks:
    result = enrich_chunk(c)  # adjust to actual signature
    preamble = result.get("preamble", result)  # adjust based on actual return shape
    word_count = len(str(preamble).split())
    print(f"ID: {c['id']}")
    print(f"Preamble ({word_count} words): {preamble}")
    print("---")

# Check cache file was written
cache_path = Path("cache/preambles.json")
if cache_path.exists():
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"\nCache file exists — {len(cache)} entries written")
else:
    print("\nWARNING: cache/preambles.json not found — caching may not be working")