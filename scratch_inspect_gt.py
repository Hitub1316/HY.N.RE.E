import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils import load_json, CHUNKS_DIR

gt = load_json(Path("data/eval/ground_truth.json"))

scene_chunks = {}
for p in CHUNKS_DIR.glob("*_scene.json"):
    for c in load_json(p):
        scene_chunks[c["id"]] = c

print("=== EXACT ENTITY QUERIES ===")
for q in gt:
    if q.get("category") == "exact_entity":
        print("Query:", q["query"])
        print("Novel:", q.get("novel"))
        for r in q.get("relevant_chunk_ids", []):
            cid = r["id"]
            chunk = scene_chunks.get(cid, {})
            text_snip = chunk.get("text", "NOT FOUND")[:200].replace("\n", " ")
            print(f"  - Chunk {cid} (grade {r['grade']}): {text_snip}...")
        print("-" * 60)

print("\n=== PARAPHRASE QUERIES ===")
for q in gt:
    if q.get("category") == "paraphrase":
        print("Query:", q["query"])
        print("Novel:", q.get("novel"))
        for r in q.get("relevant_chunk_ids", []):
            cid = r["id"]
            chunk = scene_chunks.get(cid, {})
            text_snip = chunk.get("text", "NOT FOUND")[:200].replace("\n", " ")
            print(f"  - Chunk {cid} (grade {r['grade']}): {text_snip}...")
        print("-" * 60)

print("\n=== ALL CATEGORIES OVERVIEW ===")
cat_summary = {}
for q in gt:
    cat = q.get("category", "unknown")
    cat_summary[cat] = cat_summary.get(cat, 0) + 1
for cat, count in cat_summary.items():
    print(f"  {cat:15s}: {count} queries")
