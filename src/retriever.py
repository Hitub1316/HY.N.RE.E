"""
retriever.py — Dense, BM25, Hybrid (RRF), and Hybrid+Rerank retrieval.

All retrieve_* methods return list[RankedPassage]:
    {rank, score, chunk_id, novel, chapter, text, retrieval_method}

Supports optional neighbor chunk expansion (`expand_neighbors=True`):
    returns list of evidence_window dicts:
    {target: RankedPassage, before: chunk_or_None, after: chunk_or_None}
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
        "chunk_id": chunk.get("chunk_id") or chunk.get("id", ""),
        "novel": chunk.get("novel", ""),
        "chapter": chunk.get("chapter", ""),
        "text": chunk.get("text", ""),  # Always original unmodified text
        "char_start": chunk.get("char_start", -1),
        "char_end": chunk.get("char_end", -1),
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
        self._store = None
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

    def _get_store(self):
        if self._store is None:
            self._store = load_qdrant_client()
        return self._store

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def retrieve_dense_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Embed query → cosine search in VectorStore → top-K."""
        embedder = self._get_embedder()
        store = self._get_store()
        collection = _collection_name(self.strategy)

        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits = store.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_k,
        )

        results = []
        for rank, hit in enumerate(hits):
            chunk = hit["payload"]
            chunk.setdefault("id", chunk.get("chunk_id", ""))
            results.append(_make_result(rank + 1, hit["score"], chunk, "dense"))
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
        store = self._get_store()
        collection = _collection_name(self.strategy)

        query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits = store.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_n,
        )
        results = []
        for hit in hits:
            payload = dict(hit["payload"])
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

    # ── Neighbor Expansion ────────────────────────────────────────────────────

    def expand_neighbors_for_result(self, result: dict) -> dict:
        """
        Fetch preceding and following chunks within the same chapter and novel.
        Returns:
            {
                "target": result,
                "before": preceding_chunk_or_None,
                "after": following_chunk_or_None
            }
        """
        _, all_chunks = self._get_bm25()
        novel = result["novel"]
        chapter = result["chapter"]
        target_id = result["chunk_id"]

        # Filter chunks in the same novel and chapter
        chapter_chunks = [
            c for c in all_chunks
            if c.get("novel") == novel and c.get("chapter") == chapter
        ]

        # Sort by char_start to guarantee sequential order
        chapter_chunks.sort(key=lambda x: x.get("char_start", 0))

        # Find target chunk index
        target_idx = None
        for idx, c in enumerate(chapter_chunks):
            if c.get("id") == target_id or c.get("chunk_id") == target_id:
                target_idx = idx
                break

        if target_idx is None:
            return {"target": result, "before": None, "after": None}

        before = chapter_chunks[target_idx - 1] if target_idx > 0 else None
        after = chapter_chunks[target_idx + 1] if target_idx + 1 < len(chapter_chunks) else None

        return {
            "target": result,
            "before": before,
            "after": after,
        }

    # ── Unified Public Interface ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        method: str = "hybrid_reranked",
        top_n: int = DEFAULT_TOP_N,
        top_k: int = DEFAULT_TOP_K,
        expand_neighbors: bool = False,
        **kwargs,
    ) -> list[dict]:
        """
        Dispatch to the appropriate retrieval method.

        Args:
            query: User search query.
            method: One of 'dense', 'bm25', 'hybrid', 'hybrid_reranked'.
            top_n: Candidate pool size for hybrid fusion.
            top_k: Final top-K results count.
            expand_neighbors: If True, returns evidence windows with adjacent chunks.

        Returns:
            List of RankedPassage dicts (if expand_neighbors=False)
            OR list of evidence window dicts (if expand_neighbors=True).
        """
        if method == "dense":
            results = self.retrieve_dense_only(query, top_k=top_k)
        elif method == "bm25":
            results = self.retrieve_bm25_only(query, top_k=top_k)
        elif method == "hybrid":
            results = self.retrieve_hybrid(query, top_n=top_n, top_k=top_k)
        elif method == "hybrid_reranked":
            results = self.retrieve_hybrid_reranked(query, top_n=top_n, top_k=top_k, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from ['dense', 'bm25', 'hybrid', 'hybrid_reranked'].")

        if expand_neighbors:
            return [self.expand_neighbors_for_result(r) for r in results]

        return results
