"""
indexer.py — Build dense (Qdrant via VectorStore) and lexical (BM25) indices.

One index pair per chunking strategy. Collections are named by strategy.
Embedding step embeds `contextual_text` if present, but BM25 and payloads
store original unmodified `text`.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CHUNKS_DIR, DATA_DIR, load_json
from vector_store import QdrantVectorStore

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_COLLECTION_PREFIX = "hynree"  # collections named hynree_{strategy}
BM25_PICKLE_TEMPLATE = str(DATA_DIR / "bm25_{strategy}.pkl")


# ── VectorStore helpers ───────────────────────────────────────────────────────

def _get_vector_store():
    return QdrantVectorStore()


def _collection_name(strategy: str) -> str:
    return f"{QDRANT_COLLECTION_PREFIX}_{strategy}"


def build_dense_index(chunks: list[dict], strategy: str, batch_size: int = 64) -> None:
    """
    Embed chunks and upsert into VectorStore collection.
    Embeds `contextual_text` when available; stores original `text` in payload.
    """
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    model = SentenceTransformer(EMBEDDING_MODEL)
    store = _get_vector_store()
    collection = _collection_name(strategy)

    # (Re)create collection
    if store.collection_exists(collection):
        store.delete_collection(collection)
        print(f"  Deleted existing vector collection: {collection}")

    store.create_collection(
        collection_name=collection,
        vector_size=384,
        distance="Cosine",
    )

    print(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} (using contextual_text if available) ...")
    points = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch_chunks = chunks[i: i + batch_size]
        # Text used for embedding: contextual_text if available, else original text
        batch_embed_texts = [c.get("contextual_text") or c["text"] for c in batch_chunks]
        embeddings = model.encode(batch_embed_texts, show_progress_bar=False, normalize_embeddings=True)
        for j, (chunk, emb) in enumerate(zip(batch_chunks, embeddings)):
            points.append({
                "id": i + j,
                "vector": emb.tolist(),
                "payload": {
                    "chunk_id": chunk["id"],
                    "novel": chunk["novel"],
                    "strategy": chunk["strategy"],
                    "chapter": chunk.get("chapter", ""),
                    "text": chunk["text"],  # Source of truth original text
                    "contextual_text": chunk.get("contextual_text", ""),
                    "char_start": chunk.get("char_start", -1),
                    "char_end": chunk.get("char_end", -1),
                },
            })

    store.upsert(collection_name=collection, points=points)
    print(f"  Vector collection '{collection}': {len(points)} points upserted.")


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks: list[dict], strategy: str) -> None:
    """Build BM25Okapi index on original text and pickle to disk."""
    from rank_bm25 import BM25Okapi

    print(f"  Building BM25 index for strategy='{strategy}' ...")
    # Always tokenize original text for BM25
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    out_path = Path(BM25_PICKLE_TEMPLATE.format(strategy=strategy))
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "tokenized": tokenized}, f)

    print(f"  BM25 index saved -> {out_path}")


# ── Load indices ──────────────────────────────────────────────────────────────

def load_bm25_index(strategy: str) -> tuple:
    """Return (BM25Okapi, list[Chunk]) for the given strategy."""
    path = Path(BM25_PICKLE_TEMPLATE.format(strategy=strategy))
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}. Run indexer.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def load_qdrant_client():
    """Returns QdrantVectorStore instance."""
    return _get_vector_store()


# ── Main ──────────────────────────────────────────────────────────────────────

def index_strategy(strategy: str) -> None:
    """Load chunks for `strategy` and build both vector + BM25 indices."""
    pattern = f"*_{strategy}.json"
    chunk_files = list(CHUNKS_DIR.glob(pattern))

    if not chunk_files:
        print(f"  ERROR: No chunk files found for strategy='{strategy}' in {CHUNKS_DIR}")
        print(f"  Run chunker.py / contextual_enrichment.py first.")
        return

    all_chunks = []
    for f in sorted(chunk_files):
        novel_chunks = load_json(f)
        all_chunks.extend(novel_chunks)
        print(f"  Loaded {len(novel_chunks)} chunks from {f.name}")

    print(f"  Total: {len(all_chunks)} chunks for strategy='{strategy}'")

    build_dense_index(all_chunks, strategy)
    build_bm25_index(all_chunks, strategy)
    print(f"  [OK] Indexing complete for strategy='{strategy}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant + BM25 indices.")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph", "scene", "contextual", "all"],
        default="scene",
        help="Chunking strategy to index (default: scene). Use 'all' to index everything.",
    )
    args = parser.parse_args()

    from chunker import STRATEGIES
    targets = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    print(f"=== Indexing strategies: {targets} ===\n")
    for strat in targets:
        index_strategy(strat)
