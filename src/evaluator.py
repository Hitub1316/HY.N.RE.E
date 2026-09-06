"""
evaluator.py — Precision@K, Recall@K, MRR, NDCG@10 evaluation harness.

Ground truth format (data/eval/ground_truth.json):
[
  {
    "query": "What does Elizabeth Bennet think of Mr. Darcy at first?",
    "relevant_chunks": [
      {"id": "Pride and Prejudice__scene__0042", "grade": 3},
      {"id": "Pride and Prejudice__scene__0043", "grade": 2}
    ]
  },
  ...
]

Grades: 0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant.
Binary relevance (for P@K, R@K): grade >= 2.

Usage:
    from src.evaluator import evaluate_system, run_full_ablation

    # Single system evaluation
    metrics = evaluate_system(retriever, ground_truth, method="hybrid_reranked", top_k=5)

    # Full ablation across all systems × chunking strategies
    results_df = run_full_ablation(ground_truth_path, top_k=5)
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import EVAL_DIR, load_json

BINARY_THRESHOLD = 2  # grades >= this count as relevant for P@K, R@K


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    system: str
    strategy: str
    method: str
    top_k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_10: float
    n_queries: int
    # Per-query details (not printed by default)
    per_query: list[dict] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        d = asdict(self)
        del d["per_query"]
        return d


# ── Metric implementations ────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], grade_map: dict[str, int], k: int) -> float:
    """
    NDCG@K with graded relevance.
    grade_map: {chunk_id → grade (0-3)}
    """
    def dcg(ids: list[str]) -> float:
        total = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            rel = grade_map.get(cid, 0)
            total += (2 ** rel - 1) / math.log2(i + 1)
        return total

    actual_dcg = dcg(retrieved_ids)
    # Ideal: sort all known relevant chunks by grade descending
    ideal_order = sorted(grade_map.keys(), key=lambda x: grade_map[x], reverse=True)
    ideal_dcg = dcg(ideal_order)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── Single system evaluation ──────────────────────────────────────────────────

def evaluate_system(
    retriever,
    ground_truth: list[dict],
    method: str,
    top_k: int = 5,
    ndcg_k: int = 10,
) -> MetricsResult:
    """
    Evaluate a HybridRetriever instance on the full ground truth set.

    Args:
        retriever:    A HybridRetriever instance (strategy already set).
        ground_truth: List of {query, relevant_chunks: [{id, grade}]}.
        method:       One of 'dense', 'bm25', 'hybrid', 'hybrid_reranked'.
        top_k:        K for Precision@K and Recall@K.
        ndcg_k:       K for NDCG (default 10).

    Returns:
        MetricsResult with aggregated metrics.
    """
    precision_scores = []
    recall_scores = []
    rr_scores = []
    ndcg_scores = []
    per_query = []

    for item in ground_truth:
        query = item["query"]
        relevant = item["relevant_chunks"]
        grade_map = {r["id"]: r["grade"] for r in relevant}
        binary_relevant = {r["id"] for r in relevant if r["grade"] >= BINARY_THRESHOLD}

        # Retrieve
        results = retriever.retrieve(query, method=method, top_k=max(top_k, ndcg_k))
        retrieved_ids = [r["chunk_id"] for r in results]

        # Compute metrics
        p = precision_at_k(retrieved_ids, binary_relevant, top_k)
        r = recall_at_k(retrieved_ids, binary_relevant, top_k)
        rr = reciprocal_rank(retrieved_ids, binary_relevant)
        nd = ndcg_at_k(retrieved_ids, grade_map, ndcg_k)

        precision_scores.append(p)
        recall_scores.append(r)
        rr_scores.append(rr)
        ndcg_scores.append(nd)

        per_query.append({
            "query": query,
            "precision_at_k": p,
            "recall_at_k": r,
            "rr": rr,
            "ndcg_at_10": nd,
            "retrieved_ids": retrieved_ids,
        })

    n = len(ground_truth)
    system_label = f"{retriever.strategy}/{method}"

    return MetricsResult(
        system=system_label,
        strategy=retriever.strategy,
        method=method,
        top_k=top_k,
        precision_at_k=sum(precision_scores) / n,
        recall_at_k=sum(recall_scores) / n,
        mrr=sum(rr_scores) / n,
        ndcg_at_10=sum(ndcg_scores) / n,
        n_queries=n,
        per_query=per_query,
    )


# ── Full ablation runner ──────────────────────────────────────────────────────

# Maps (display_name, strategy, method) for the 5 ablation systems
ABLATION_SYSTEMS = [
    ("BM25 only",                  "scene",       "bm25"),
    ("Dense only",                 "scene",       "dense"),
    ("Hybrid (RRF)",               "scene",       "hybrid"),
    ("Hybrid + Rerank",            "scene",       "hybrid_reranked"),
    ("Contextual Hybrid + Rerank", "contextual",  "hybrid_reranked"),
]

CHUNKING_STRATEGIES = [
    ("Fixed token",           "fixed"),
    ("Paragraph",             "paragraph"),
    ("Scene-aware",           "scene"),
    ("Contextual scene-aware","contextual"),
]


def run_full_ablation(
    ground_truth_path: Path | str,
    top_k: int = 5,
    ndcg_k: int = 10,
) -> "pd.DataFrame":
    """
    Run the full 5-system ablation and return a pandas DataFrame.

    Requires Qdrant and BM25 indices to already be built (run indexer.py first).
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retriever import HybridRetriever

    ground_truth = load_json(Path(ground_truth_path))
    rows = []

    for display_name, strategy, method in ABLATION_SYSTEMS:
        print(f"  Evaluating: {display_name} ...")
        retriever = HybridRetriever(strategy=strategy)
        metrics = evaluate_system(retriever, ground_truth, method, top_k, ndcg_k)
        row = {
            "System": display_name,
            f"Recall@{top_k}": round(metrics.recall_at_k, 4),
            "MRR": round(metrics.mrr, 4),
            f"NDCG@{ndcg_k}": round(metrics.ndcg_at_10, 4),
        }
        rows.append(row)
        print(f"    Recall@{top_k}={metrics.recall_at_k:.4f}  MRR={metrics.mrr:.4f}  NDCG@{ndcg_k}={metrics.ndcg_at_10:.4f}")

    return pd.DataFrame(rows).set_index("System")


def run_chunking_ablation(
    ground_truth_path: Path | str,
    method: str = "hybrid_reranked",
    top_k: int = 5,
) -> "pd.DataFrame":
    """
    Run chunking strategy ablation (fixed method, vary strategy).
    Returns pandas DataFrame.
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retriever import HybridRetriever

    ground_truth = load_json(Path(ground_truth_path))
    rows = []

    for display_name, strategy in CHUNKING_STRATEGIES:
        print(f"  Evaluating chunking: {display_name} ...")
        retriever = HybridRetriever(strategy=strategy)
        metrics = evaluate_system(retriever, ground_truth, method, top_k)
        rows.append({
            "Chunking Strategy": display_name,
            f"Recall@{top_k}": round(metrics.recall_at_k, 4),
            "MRR": round(metrics.mrr, 4),
        })

    return pd.DataFrame(rows).set_index("Chunking Strategy")
