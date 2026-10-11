import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils import load_json, CHUNKS_DIR
from retriever import HybridRetriever, _tokenize
from indexer import load_bm25_index

query = "What does Elizabeth Bennet think of Mr. Darcy at the Netherfield ball?"

print(f"=== DIAGNOSTIC INVESTIGATION FOR QUERY: '{query}' ===\n")

# Step 1: Check Chapter XVIII scene chunks
pp_chunks_path = CHUNKS_DIR / "Pride_and_Prejudice_scene.json"
chunks = load_json(pp_chunks_path)

ch18_chunks = [c for c in chunks if "CHAPTER XVIII" in c.get("chapter", "").upper()]

print(f"Step 1: Total scene chunks in Pride and Prejudice: {len(chunks)}")
print(f"        Total Chapter XVIII scene chunks: {len(ch18_chunks)} (IDs: {ch18_chunks[0]['id']} to {ch18_chunks[-1]['id']})")
print("-" * 75)

# Initialize single retriever instance
retriever = HybridRetriever(strategy="scene")

# Step 2a: BM25 Scores
bm25, bm25_chunks = load_bm25_index("scene")
tokenized_q = _tokenize(query)
bm25_scores = bm25.get_scores(tokenized_q)
bm25_pp_indices = [i for i, c in enumerate(bm25_chunks) if c["novel"] == "Pride and Prejudice"]
bm25_pp_scores = [(bm25_scores[i], bm25_chunks[i]["id"], bm25_chunks[i]["chapter"], bm25_chunks[i]["text"]) for i in bm25_pp_indices]
bm25_pp_scores.sort(key=lambda x: x[0], reverse=True)

print("\nStep 2a: Top-5 BM25 Scene Chunks (Pride and Prejudice only):")
for rank, (score, cid, chap, text) in enumerate(bm25_pp_scores[:5], 1):
    print(f"  Rank {rank:2d} | BM25 Score: {score:7.4f} | {cid} | {chap}")

ch18_bm25_ranks = [(rank+1, score, cid) for rank, (score, cid, chap, text) in enumerate(bm25_pp_scores) if "CHAPTER XVIII" in chap.upper()]
print(f"  -> Chapter XVIII highest BM25 rank: Rank {ch18_bm25_ranks[0][0]} (Score: {ch18_bm25_ranks[0][1]:.4f}, ID: {ch18_bm25_ranks[0][2]})")

# Step 2b: Dense Scores
embedder = retriever._get_embedder()
store = retriever._get_store()
query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
hits = store.search(collection_name="hynree_scene", query_vector=query_vec, limit=566)
dense_pp_hits = [h for h in hits if h["payload"].get("novel") == "Pride and Prejudice"]

print("\nStep 2b: Top-5 Dense Scene Chunks (Pride and Prejudice only):")
for rank, hit in enumerate(dense_pp_hits[:5], 1):
    payload = hit["payload"]
    print(f"  Rank {rank:2d} | Dense Score: {hit['score']:7.4f} | {payload['chunk_id']} | {payload.get('chapter')}")

ch18_dense_ranks = [(rank+1, hit['score'], hit['payload']['chunk_id']) for rank, hit in enumerate(dense_pp_hits) if "CHAPTER XVIII" in hit['payload'].get('chapter', '').upper()]
print(f"  -> Chapter XVIII highest Dense rank: Rank {ch18_dense_ranks[0][0]} (Score: {ch18_dense_ranks[0][1]:.4f}, ID: {ch18_dense_ranks[0][2]})")

# Step 2c: Hybrid + Rerank Scores
hybrid_reranked_results = retriever.retrieve(query, method="hybrid_reranked", top_k=10, expand_neighbors=False)
print("\nStep 2c: Top-10 Hybrid + Rerank Results (Overall):")
for r in hybrid_reranked_results:
    print(f"  Rank {r['rank']:2d} | Rerank Score: {r['score']:7.4f} | {r['chunk_id']} | {r['chapter']} ({r['novel']})")

# Step 3: Test with expand_neighbors=True
print("\nStep 3: Testing expand_neighbors=True for Top-5 Hybrid+Rerank Results:")
evidence_results = retriever.retrieve(query, method="hybrid_reranked", top_k=5, expand_neighbors=True)
for idx, ev in enumerate(evidence_results, 1):
    target = ev["target"]
    before = ev["before"]
    after = ev["after"]
    print(f"  Window #{idx}: Target={target['chunk_id']} ({target['chapter']})")
    if before:
        print(f"    Before: {before['id']} ({before['chapter']}) -> {before['text'][:80]}...")
    else:
        print("    Before: None")
    print(f"    Target: {target['chunk_id']} ({target['chapter']}) -> {target['text'][:80]}...")
    if after:
        print(f"    After:  {after['id']} ({after['chapter']}) -> {after['text'][:80]}...")
    else:
        print("    After:  None")
    print("-" * 50)
