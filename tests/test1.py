"""
test1.py — Smoke test: fixed, paragraph, and scene chunking strategies on all 3 novels.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chunker import chunk

NOVELS = ["Pride and Prejudice", "Jane Eyre", "Little Women"]

for NOVEL in NOVELS:
    NOVEL_PATH = Path("novels") / f"{NOVEL}.txt"

    if not NOVEL_PATH.exists():
        print(f"ERROR: {NOVEL_PATH} not found. Run `python src/corpus_prep.py` first.")
        continue

    text = NOVEL_PATH.read_text(encoding="utf-8")
    print(f"\n\n########################################################################")
    print(f"NOVEL: {NOVEL} — {len(text.split()):,} words")
    print(f"########################################################################")

    for strat in ["fixed", "paragraph", "scene"]:
        chunks = chunk(text, strategy=strat, novel=NOVEL)
        print(f"======================================================================")
        print(f"STRATEGY: {strat.upper()} | Total chunks: {len(chunks)}")
        print(f"Schema keys: {list(chunks[0].keys()) if chunks else 'N/A'}")
        print("----------------------------------------------------------------------")
        for c in chunks[:2]:
            print(json.dumps(c, indent=2, ensure_ascii=False).encode('utf-8', errors='replace').decode('utf-8'))
            print("---")