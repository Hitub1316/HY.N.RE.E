"""
retriever.py — Dense, BM25, Hybrid (RRF), and Hybrid+Rerank retrieval.

All retrieve_* methods return list[RankedPassage]:
    {rank, score, chunk_id, novel, chapter, text, retrieval_method}

Usage:
    from src.retriever import HybridRetriever

    r = HybridRetriever(strategy="scene")
    results = r.retrieve_hybrid_reranked("What does Mr. Darcy say to Elizabeth?", top_k=5)
    for p in results:
        print(p["rank"], p["score"], p["text"][:120])
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexer import load_bm25_index, load_qdrant_client, _collection_name, _tokenize, EMBEDDING_MODEL

RRF_K = 60          # RRF constant
DEFAULT_TOP_N = 20  # candidates fetched from each retriever before fusion
DEFAULT_TOP_K = 5   # final results returned to caller
RERANK_POOL = 50    # top-N from RRF passed to cross-encoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Result schema ─────────────────────────────────────────────────────────────

def _make_result(rank: int, score: float, chunk: dict, method: str) -> dict:
    return {
        "rank": rank,
        "score": score,
        "chunk_id": chunk["id"] if "id" in chunk else chunk.get("chunk_id", ""),
        "novel": chunk.get("novel", ""),
        "chapter": chunk.get("chapter", ""),
        "text": chunk.get("text", ""),
        "retrieval_method": method,
    }


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunks.
    Each list element must have a 'chunk_id' key.
    Returns re-ranked list of chunks with 'rrf_score'.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank_idx, item in enumerate(ranked):
            cid = item.get("chunk_id") or item.get("id", "")
            score = 1.0 / (k + rank_idx + 1)
            scores[cid] = scores.get(cid, 0.0) + score
            if cid not in chunk_map:
                chunk_map[cid] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = scores[cid]
        fused.append(item)
    return fused


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Retrieval interface for a given chunking strategy.

    Args:
        strategy: One of 'fixed', 'paragraph', 'scene', 'contextual'.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self._embedding_model = None
        self._bm25 = None
        self._bm25_chunks = None
        self._qdrant = None
        self._reranker = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _get_bm25(self):
        if self._bm25 is None:
            self._bm25, self._bm25_chunks = load_bm25_index(self.strategy)
        return self._bm25, self._bm25_chunks

    def _get_qdrant(self):
        if self._qdrant is None:
            self._qdrant = load_qdrant_client()
        return self._qdrant

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def retrieve_dense_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Embed query → cosine search in Qdrant → top-K."""
        embedder = self._get_embedder()
        qdrant = self._get_qdrant()
        collection = _collection_name(self.strategy)

        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits = qdrant.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for rank, hit in enumerate(hits):
            chunk = hit.payload
            chunk.setdefault("id", chunk.get("chunk_id", ""))
            results.append(_make_result(rank + 1, hit.score, chunk, "dense"))
        return results

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def retrieve_bm25_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Tokenize query → BM25 → top-K."""
        bm25, chunks = self._get_bm25()
        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = chunks[idx]
            results.append(_make_result(rank + 1, float(scores[idx]), chunk, "bm25"))
        return results

    # ── Hybrid (RRF) ──────────────────────────────────────────────────────────

    def _dense_candidates(self, query: str, top_n: int) -> list[dict]:
        """Return top-N dense results as lightweight dicts with chunk_id."""
        embedder = self._get_embedder()
        qdrant = self._get_qdrant()
        collection = _collection_name(self.strategy)

        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits = qdrant.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_n,
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = dict(hit.payload)
            payload["chunk_id"] = payload.get("chunk_id", payload.get("id", ""))
            results.append(payload)
        return results

    def _bm25_candidates(self, query: str, top_n: int) -> list[dict]:
        """Return top-N BM25 results as lightweight dicts with chunk_id."""
        bm25, chunks = self._get_bm25()
        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            chunk = dict(chunks[idx])
            chunk["chunk_id"] = chunk.get("id", chunk.get("chunk_id", ""))
            results.append(chunk)
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Dense + BM25 → RRF fusion → top-K."""
        dense_cands = self._dense_candidates(query, top_n)
        bm25_cands = self._bm25_candidates(query, top_n)
        fused = _rrf_fuse([dense_cands, bm25_cands])[:top_k]

        results = []
        for rank, item in enumerate(fused):
            results.append(_make_result(rank + 1, item["rrf_score"], item, "hybrid_rrf"))
        return results

    # ── Hybrid + Rerank ───────────────────────────────────────────────────────

    def retrieve_hybrid_reranked(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        top_k: int = DEFAULT_TOP_K,
        rerank_pool: int = RERANK_POOL,
    ) -> list[dict]:
        """Dense + BM25 → RRF → cross-encoder rerank → top-K."""
        dense_cands = self._dense_candidates(query, top_n)
        bm25_cands = self._bm25_candidates(query, top_n)
        fused = _rrf_fuse([dense_cands, bm25_cands])[:rerank_pool]

        if not fused:
            return []

        reranker = self._get_reranker()
        pairs = [(query, item["text"]) for item in fused]
        cross_scores = reranker.predict(pairs)

        import numpy as np
        ranked_indices = np.argsort(cross_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            item = fused[idx]
            results.append(_make_result(rank + 1, float(cross_scores[idx]), item, "hybrid_reranked"))
        return results

    # ── Convenience ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, method: str = "hybrid_reranked", **kwargs) -> list[dict]:
        """
        Dispatch to the appropriate retrieval method.

        Args:
            method: One of 'dense', 'bm25', 'hybrid', 'hybrid_reranked'.
        """
        dispatch = {
            "dense": self.retrieve_dense_only,
            "bm25": self.retrieve_bm25_only,
            "hybrid": self.retrieve_hybrid,
            "hybrid_reranked": self.retrieve_hybrid_reranked,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}.")
        return dispatch[method](query, **kwargs)
