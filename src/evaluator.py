"""
evaluator.py — Precision@K, Recall@K, MRR, NDCG@10 evaluation harness with per-category breakdown.

Ground truth format (data/eval/ground_truth.json):
[
  {
    "query": "What does Elizabeth Bennet think of Mr. Darcy at first?",
    "category": "semantic",
    "relevant_chunk_ids": [
      {"id": "Pride and Prejudice__scene__0042", "grade": 3},
      {"id": "Pride and Prejudice__scene__0043", "grade": 2}
    ]
  },
  ...
]

Grades: 0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant.
Binary relevance (for P@K, R@K): grade >= 2.
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import EVAL_DIR, load_json, save_json

BINARY_THRESHOLD = 2  # grades >= this count as relevant for P@K, R@K


def _normalize_id(cid: str) -> str:
    """Normalize chunk ID across strategies (e.g. Novel__scene__0042 -> Novel__0042)."""
    parts = cid.split("__")
    if len(parts) == 3:
        return f"{parts[0]}__{parts[2]}"
    return cid


# ── Metric implementations ────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = [_normalize_id(r) for r in retrieved_ids[:k]]
    norm_relevant = {_normalize_id(r) for r in relevant_ids}
    hits = sum(1 for r in top_k if r in norm_relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = [_normalize_id(r) for r in retrieved_ids[:k]]
    norm_relevant = {_normalize_id(r) for r in relevant_ids}
    hits = sum(1 for r in top_k if r in norm_relevant)
    return hits / len(norm_relevant) if norm_relevant else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    norm_relevant = {_normalize_id(r) for r in relevant_ids}
    for rank, cid in enumerate(retrieved_ids, start=1):
        if _normalize_id(cid) in norm_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], grade_map: dict[str, int], k: int) -> float:
    """NDCG@K with graded relevance."""
    norm_grade_map = {_normalize_id(cid): grade for cid, grade in grade_map.items()}

    def dcg(ids: list[str]) -> float:
        total = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            norm_cid = _normalize_id(cid)
            rel = norm_grade_map.get(norm_cid, 0)
            total += (2 ** rel - 1) / math.log2(i + 1)
        return total

    actual_dcg = dcg(retrieved_ids)
    ideal_order = sorted(norm_grade_map.keys(), key=lambda x: norm_grade_map[x], reverse=True)
    ideal_dcg = dcg(ideal_order)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _avg_metrics(metrics_list: list[dict]) -> dict:
    """Helper to compute mean metrics over a list of query metric dicts."""
    if not metrics_list:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0, "n_queries": 0}
    n = len(metrics_list)
    return {
        "precision_at_k": round(sum(m["p"] for m in metrics_list) / n, 4),
        "recall_at_k": round(sum(m["r"] for m in metrics_list) / n, 4),
        "mrr": round(sum(m["rr"] for m in metrics_list) / n, 4),
        "ndcg_at_10": round(sum(m["nd"] for m in metrics_list) / n, 4),
        "n_queries": n,
    }


# ── Single system evaluation ──────────────────────────────────────────────────

def evaluate_system(
    retriever,
    ground_truth: list[dict],
    method: str,
    top_k: int = 5,
    ndcg_k: int = 10,
) -> dict:
    """
    Evaluate a HybridRetriever instance on ground truth with per-category breakdown.
    """
    per_query = []
    by_category_queries: dict[str, list[dict]] = {}

    for item in ground_truth:
        query = item["query"]
        category = item.get("category", "uncategorized")
        relevant = item.get("relevant_chunk_ids") or item.get("relevant_chunks", [])
        grade_map = {r["id"]: r["grade"] for r in relevant}
        binary_relevant = {r["id"] for r in relevant if r["grade"] >= BINARY_THRESHOLD}

        results = retriever.retrieve(query, method=method, top_k=max(top_k, ndcg_k))
        retrieved_ids = [r["chunk_id"] for r in results]

        p = precision_at_k(retrieved_ids, binary_relevant, top_k)
        r = recall_at_k(retrieved_ids, binary_relevant, top_k)
        rr = reciprocal_rank(retrieved_ids, binary_relevant)
        nd = ndcg_at_k(retrieved_ids, grade_map, ndcg_k)

        q_metric = {
            "query": query,
            "category": category,
            "p": p,
            "r": r,
            "rr": rr,
            "nd": nd,
            "retrieved_ids": retrieved_ids,
        }
        per_query.append(q_metric)

        if category not in by_category_queries:
            by_category_queries[category] = []
        by_category_queries[category].append(q_metric)

    overall_metrics = _avg_metrics(per_query)
    by_category_metrics = {
        cat: _avg_metrics(q_list)
        for cat, q_list in sorted(by_category_queries.items())
    }

    system_label = f"{retriever.strategy}/{method}"

    return {
        "system": system_label,
        "strategy": retriever.strategy,
        "method": method,
        "overall": overall_metrics,
        "by_category": by_category_metrics,
    }


# ── Full ablation runner ──────────────────────────────────────────────────────

ABLATION_SYSTEMS = [
    ("BM25 only",                  "scene",       "bm25"),
    ("Dense only",                 "scene",       "dense"),
    ("Hybrid (RRF)",               "scene",       "hybrid"),
    ("Hybrid + Rerank",            "scene",       "hybrid_reranked"),
    ("Contextual Hybrid + Rerank", "contextual",  "hybrid_reranked"),
]


def run_full_ablation(
    ground_truth_path: Path | str = EVAL_DIR / "ground_truth.json",
    top_k: int = 5,
    ndcg_k: int = 10,
) -> dict:
    """
    Run full 5-system evaluation ablation across overall and all query categories.
    """
    import pandas as pd
    from retriever import HybridRetriever

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        from build_ground_truth import generate_ground_truth
        print(f"Ground truth file not found at {gt_path}. Generating ground truth...")
        ground_truth = generate_ground_truth()
    else:
        ground_truth = load_json(gt_path)

    print(f"\n==========================================================================")
    print(f"RUNNING FULL ABLATION ({len(ground_truth)} Queries, 5 Systems)")
    print(f"==========================================================================")

    all_system_results = {}
    overall_rows = []

    for display_name, strategy, method in ABLATION_SYSTEMS:
        print(f"Evaluating System: {display_name:28s} (strategy={strategy}, method={method})...")
        retriever = HybridRetriever(strategy=strategy)
        sys_res = evaluate_system(retriever, ground_truth, method, top_k, ndcg_k)
        all_system_results[display_name] = sys_res

        ov = sys_res["overall"]
        overall_rows.append({
            "System": display_name,
            f"Recall@{top_k}": ov["recall_at_k"],
            "MRR": ov["mrr"],
            f"NDCG@{ndcg_k}": ov["ndcg_at_10"],
            f"P@{top_k}": ov["precision_at_k"],
        })

    # Build per-category Summary Table (Recall@5 & MRR by Category)
    categories = sorted({q.get("category", "") for q in ground_truth if q.get("category")})
    cat_rows = []
    for display_name, _, _ in ABLATION_SYSTEMS:
        sys_cat = all_system_results[display_name]["by_category"]
        row = {"System": display_name}
        for cat in categories:
            cat_metrics = sys_cat.get(cat, {"recall_at_k": 0.0, "mrr": 0.0})
            row[f"{cat} (R@{top_k})"] = cat_metrics.get("recall_at_k", 0.0)
            row[f"{cat} (MRR)"] = cat_metrics.get("mrr", 0.0)
        cat_rows.append(row)

    df_overall = pd.DataFrame(overall_rows).set_index("System")
    df_by_category = pd.DataFrame(cat_rows).set_index("System")

    print("\n" + "=" * 70)
    print("OVERALL ABLATION METRICS")
    print("=" * 70)
    print(df_overall.to_string())

    print("\n" + "=" * 70)
    print("PER-CATEGORY RECALL & MRR BREAKDOWN")
    print("=" * 70)
    print(df_by_category.to_string())

    output_payload = {
        "overall_summary": df_overall.to_dict(orient="index"),
        "by_category_summary": df_by_category.to_dict(orient="index"),
        "raw_results": all_system_results,
    }
    save_json(output_payload, EVAL_DIR / "ablation_results.json")
    print(f"\n  Saved detailed ablation results -> {EVAL_DIR / 'ablation_results.json'}\n")

    return output_payload


if __name__ == "__main__":
    run_full_ablation()
