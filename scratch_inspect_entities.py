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

print("=== DETAILED INSPECTION: EXACT ENTITY QUERIES & SOURCE PASSAGES ===\n")
for idx, q in enumerate(gt, 1):
    if q.get("category") == "exact_entity":
        print(f"[{idx}] Query: {q['query']}")
        print(f"    Novel: {q.get('novel')}")
        for r in q.get("relevant_chunk_ids", []):
            cid = r["id"]
            chunk = scene_chunks.get(cid, {})
            text = chunk.get("text", "NOT FOUND").replace("\n", " ")
            print(f"    -> Labeled Chunk: {cid} (grade {r['grade']})")
            print(f"       Passage Text (first 250 chars): {text[:250]}...")
        print("-" * 75)
