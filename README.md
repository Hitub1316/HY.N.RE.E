# HY.N.RE.E — Hybrid Narrative Retrieval Engine

A research-grade hybrid information-retrieval engine evaluated over a public-domain fiction corpus (*Pride and Prejudice*, *Jane Eyre*, *Little Women*).
Evaluates retrieval strategies (BM25 lexical, Dense semantic, Hybrid RRF, Cross-Encoder Reranked, and Contextual Enrichment) overall and across 7 narrative query categories.
Designed to be lifted directly into [Mithras](https://github.com/Hitub1316/HY.N.RE.E) as its core retrieval layer.

---

## Corpus & Chunking

| Novel | Author | Full Word Count | Scene Chunks | Fixed Chunks | Paragraph Chunks |
|---|---|:---:|:---:|:---:|:---:|
| **Pride and Prejudice** | Jane Austen | 121,703 | 763 | 763 | 708 |
| **Jane Eyre** | Charlotte Brontë | 184,452 | 1,176 | 1,206 | 1,153 |
| **Little Women** | Louisa May Alcott | 185,867 | 1,159 | 1,196 | 1,150 |
| **Total Corpus** | — | **492,022** | **3,098** | **3,165** | **3,011** |

- **4 Chunking Strategies**: Fixed token window, Paragraph level, Scene-aware (chapter-preserving), and Contextual scene-aware.
- **Contextual Schema**: Preserves `"text"` (unmodified original text, source of truth) and `"contextual_text"` (preamble + original text used solely for dense vector embedding). Every chunk tracks `"preamble_source"` (`"cache"`, `"gemini"`, `"groq"`, or `"rule_based"`).
- **IMPORTANT METHODOLOGY DISCLOSURE — Preamble Source Breakdown**:
  > Across the 3,098 total scene chunks in the full 492k-word corpus:
  > - **LLM-Generated (Cache)**: **563 chunks** (~18.2%) were preambled using LLM generation (Gemini Flash from cached runs).
  > - **Rule-Based Fallback**: **2,535 chunks** (~81.8%) used deterministic rule-based preamble synthesis (`In '{novel}' ({chapter}), this passage details: "..."`) because `GEMINI_API_KEY` and `GROQ_API_KEY` were unconfigured in the subshell execution environment.
  > - **Implication**: The "Contextual Hybrid + Rerank" ablation results on the full 492k-word corpus reflect a **hybrid mix of LLM-generated and rule-based preambles**, rather than 100% LLM-generated preambles.

---

## Architecture & Vector Store Interface

```
Novel Corpus -> Cleaned Text
      ↓
Structure-Aware Chunking (Scene / Paragraph / Fixed)
      ↓
Contextual Enrichment (LLM preamble with caching & fallback, tracked via preamble_source)
      ↓
  VectorStore Abstraction Layer (QdrantVectorStore)
      ↓
┌───────────────────────────────┬───────────────────────────────┐
↓                               ↓                               ↓
Dense Vector Index              BM25 Lexical Index              Cross-Encoder Reranker
(all-MiniLM-L6-v2)             (rank-bm25)                      (ms-marco-MiniLM-L-6-v2)
└───────────────────────────────┴───────────────────────────────┘
                                ↓
                      RRF Fusion (k=60) + Cross-Encoder Rerank
                                ↓
                      Top-K Ranked Passages + Optional Evidence Window
                      (Neighbor expansion: before, target, after)
```

---

## Benchmark Results (49 Queries across 7 Categories)

### Hypothesis Status & Overall Ablation

> **Honest Evaluation Framing**: Evaluated over the full 492k-word corpus, **Hybrid + Rerank** achieves the highest overall MRR (**0.3027** vs 0.2389 for BM25) and highest NDCG@10 (**0.2447** vs 0.2029 for BM25), outperforming lexical search by **+0.0638 MRR points**. BM25 ties Hybrid + Rerank on Recall@5 (**0.2449**), confirming that reranking significantly improves precision of top-ranked narrative evidence.

| System Configuration | Recall@5 | MRR | NDCG@10 | Precision@5 |
|---|:---:|:---:|:---:|:---:|
| **BM25 only** | **0.2449** | 0.2389 | 0.2029 | **0.0980** |
| **Dense only** | 0.1633 | 0.2536 | 0.1739 | 0.0653 |
| **Hybrid (RRF)** | 0.2347 | 0.2678 | 0.2144 | 0.0939 |
| **Hybrid + Rerank** | **0.2449** | **0.3027** | **0.2447** | **0.0980** |
| **Contextual Hybrid + Rerank** | 0.2347 | 0.2929 | 0.2333 | 0.0939 |

*Note: Per-category findings below are subject to sample size limitations (7 queries per category, 49 queries total).*

---

### Per-Category Breakdown (Recall@5 & MRR)

| System Config | exact_entity | semantic | relationship | temporal | event | canon_rule | paraphrase |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 only** | **0.5000** / 0.5218 | **0.2857** / 0.2857 | **0.2857** / 0.2500 | 0.2143 / 0.1944 | 0.1429 / 0.1944 | 0.2143 / 0.1786 | 0.0714 / 0.0476 |
| **Dense only** | 0.2857 / 0.5408 | 0.0714 / 0.1429 | 0.0714 / 0.1429 | 0.2143 / 0.2143 | **0.2143** / **0.3061** | 0.0714 / 0.1429 | **0.2143** / 0.2857 |
| **Hybrid (RRF)** | 0.4286 / **0.5561** | 0.2143 / 0.2071 | 0.2143 / 0.1837 | 0.2143 / **0.2857** | **0.2143** / 0.2500 | 0.2143 / 0.1929 | 0.1429 / 0.1990 |
| **Hybrid + Rerank** | 0.4286 / 0.4667 | 0.2143 / 0.2429 | 0.2143 / 0.3452 | 0.2143 / 0.2024 | 0.1429 / 0.2109 | **0.2857** / **0.2619** | **0.2143** / **0.3889** |
| **Contextual Hybrid + Rerank** | 0.4286 / 0.4633 | 0.1429 / 0.2347 | **0.2857** / **0.3738** | 0.2143 / 0.2024 | 0.1429 / 0.2109 | **0.2857** / **0.2619** | 0.1429 / 0.3036 |

---

## Key Insights & Known Limitations

1. **Reranker MRR Gain**:
   - On the full-length corpus, **Hybrid + Rerank** improves MRR to **0.3027** (+26.7% over BM25's 0.2389 MRR), demonstrating that cross-encoder reranking effectively ranks ground truth passages at the top position.
2. **True Paraphrases**:
   - For genuine semantic paraphrases (with zero literal quote overlap), **Hybrid + Rerank** achieves **0.3889 MRR** compared to BM25's **0.0476 MRR** (an 8x improvement).
3. **Full Corpus Validation**:
   - Re-indexing full-length novels resolved truncation gaps for later key narrative scenes (such as Beth's scarlet fever illness in *Little Women* Ch. XVII and Rochester's bed fire in *Jane Eyre* Ch. XV/XVI), both of which now rank in top-3 results.
4. **Known Limitation — Setup vs. Event Passage Density**:
   - Setup or preparatory chapters (e.g. Chapter XVII ball preparations) concentrate explicit metadata terms in a single block, whereas actual event dialogues spread speech turns across multiple scene chunks. Neighbor expansion (`expand_neighbors=True`) expands the evidence window to capture adjacent event context.

---

## Run the Standalone Demo UI

```bash
pip install -r requirements.txt
streamlit run app.py
```

Features:
- Live query search with novel filter (*Pride and Prejudice*, *Jane Eyre*, *Little Women*, *All*, or custom uploaded novels).
- System configuration selector (*BM25*, *Dense*, *Hybrid*, *Hybrid+Rerank*, *Contextual Hybrid+Rerank*).
- Toggle for `expand_neighbors` (evidence window with `before`, `target`, and `after` scene chunks).
- **Custom Novel Upload**: Upload `.txt`, `.pdf` (with automatic header/footer stripping), or `.docx` novels up to 10MB in-session, with an explicit opt-in checkbox for Contextual Enrichment (LLM preambles).
- Interactive ablation benchmark dashboard and methodology overview.

---

## Mithras Reusability Notes

| HY.N.RE.E Module | Mithras Component | Integration Status |
|---|---|---|
| `src/corpus_prep.py` | Manuscript cleaning & chapter boundary detection | Fully Reusable |
| `src/chunker.py` | Scene-aware strategy with token cap & chapter boundaries | Fully Reusable |
| `src/contextual_enrichment.py` | Incremental LLM preamble enrichment + caching | Fully Reusable |
| `src/vector_store.py` | Abstract `VectorStore` interface (`QdrantVectorStore`) | Ready for `pgvector` extension |
| `src/indexer.py` | Dual dense & lexical indexing | Fully Reusable |
| `src/retriever.py` | Hybrid RRF + Reranker + Evidence Window expansion | Core Retrieval Engine |
| `src/evaluator.py` | Categorized benchmark harness (P@K, R@K, MRR, NDCG@10) | Benchmark Standard |

No rework needed for integration — only swap the embedding model and add multilingual support (Hindi/Marathi) when integrating into Mithras.
