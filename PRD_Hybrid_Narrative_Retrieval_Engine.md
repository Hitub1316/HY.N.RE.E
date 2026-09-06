# PRD: HY.N.RE.E

**Author:** Hitu Bharal
**Status:** Draft
**Type:** Standalone mini-project → future Mithras retrieval layer

---

## 1. Objective

Build and empirically evaluate a hybrid information-retrieval system over a niche fiction corpus. Prove which retrieval strategy — lexical, dense, hybrid, reranked, contextual — performs best on narrative text.

---

## 2. Problem Statement

Standard RAG demos ("chat with PDFs") lack retrieval rigor. Dense-only retrieval misses exact-term queries (character names, locations, exact phrasing). Lexical-only retrieval misses paraphrased or semantic queries. No single method wins on all query types.

This project builds a retrieval system that combines both, measures the improvement quantitatively, and produces a component directly reusable as Mithras's retrieval layer.

---

## 3. Goals

- Build a working hybrid retrieval pipeline: dense + BM25 + RRF fusion + cross-encoder reranking.
- Add Contextual Retrieval (contextual chunking) to measure accuracy uplift.
- Run chunking-strategy ablation (fixed vs paragraph vs scene-aware vs contextual).
- Produce a ground-truth evaluation set and measure Precision@K, Recall@K, MRR, NDCG.
- Package as a demo showing query → ranked passages, with metrics displayed.
- Design system so it can be lifted directly into Mithras later, with no rework.

## 4. Non-Goals

- No ColBERT / late-interaction retrieval (v1).
- No query expansion / HyDE (v1).
- No production deployment, auth, or multi-user support.
- No LLM-based answer generation — retrieval only, not full RAG chat.

---

## 5. Target Corpus

Public-domain novels (Project Gutenberg):
- Pride and Prejudice
- Jane Eyre
- Little Women

Chosen for character/relationship/event density — mirrors Mithras's actual use case. Little Women replaces Frankenstein: it is longer, more character-dense, and provides richer relationship-query test coverage (four protagonists + ensemble cast vs. Frankenstein's narrower cast).

---

## 6. System Architecture

```
Novel corpus
     ↓
Structure-aware chunking (scene/paragraph level)
     ↓
Contextual enrichment (LLM-generated chunk context, prepended)
     ↓
     ┌──────────────┬──────────────┐
     ↓              ↓
Dense retrieval   BM25 retrieval
(sentence-transformers +   (rank_bm25)
 Qdrant/pgvector)
     ↓              ↓
     └──────┬───────┘
            ↓
      RRF Fusion (k=60)
            ↓
    Cross-Encoder Reranker
            ↓
      Top-K evidence passages
            ↓
       Evaluation harness
```

---

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | System ingests raw novel text and splits into chunks |
| FR2 | System generates contextual preamble per chunk (cached LLM call) |
| FR3 | System embeds chunks (local sentence-transformers model) and stores in vector DB |
| FR4 | System builds BM25 index over same chunks |
| FR5 | Given a query, system retrieves top-N from dense and top-N from BM25 |
| FR6 | System fuses both ranked lists via RRF |
| FR7 | System reranks fused top-K via cross-encoder |
| FR8 | System returns final ranked passages with scores |
| FR9 | System supports swapping chunking strategy (fixed/paragraph/scene/contextual) for ablation |
| FR10 | System computes Precision@K, Recall@K, MRR, NDCG@10 against ground-truth query set |

---

## 8. Tech Stack

| Layer | Choice |
|---|---|
| Embeddings | sentence-transformers (local, no API cost) |
| Vector store | Qdrant (or pgvector — matches Mithras stack) |
| Lexical retrieval | rank_bm25 |
| Fusion | Reciprocal Rank Fusion, k=60 |
| Reranker | Cross-encoder (ms-marco-MiniLM or similar) |
| Contextual chunk generation | LLM API call, cached |
| Evaluation | Custom harness (Precision/Recall/MRR/NDCG) |
| Demo | Notebook or minimal UI |

---

## 9. Evaluation Plan

Ground-truth set: hand-built queries spanning character facts, relationships, events, paraphrased references, exact-term lookups.

Ablation matrix to fill:

| System | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|
| BM25 only | | | |
| Dense only | | | |
| Hybrid (RRF) | | | |
| Hybrid + Rerank | | | |
| Contextual Hybrid + Rerank | | | |

Chunking ablation (secondary):

| Chunking strategy | Recall@5 | MRR |
|---|---|---|
| Fixed token | | |
| Paragraph | | |
| Scene-aware | | |
| Contextual scene-aware | | |

---

## 10. Success Criteria

- Hybrid + rerank beats both single-method baselines on Recall@5 and NDCG@10.
- Contextual chunking shows measurable uplift over non-contextual.
- Full pipeline runs end-to-end on all 3 novels without manual intervention.
- README documents methodology, results table, and reusability path into Mithras.

---

## 11. Timeline

| Phase | Time |
|---|---|
| Corpus prep + chunking + dense/BM25 retrieval | Weekend 1, Day 1 |
| RRF fusion + cross-encoder rerank | Weekend 1, Day 2 |
| Contextual retrieval + chunking ablation | Weekend 2, Day 1 |
| Evaluation harness + ground-truth set + results | Weekend 2, Day 2 |

Total: 2 weekends.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Ground-truth query set too small for reliable metrics | Aim for 30–50 queries min across query types |
| Cross-encoder too slow on full corpus | Rerank only top-50 fused candidates, not full set |
| Contextual chunk generation costs add up | Use prompt caching, keep context to 50–100 tokens |
| Novel domain too narrow to generalize claims | Frame as domain-specific IR study, not general claim |

---

## 13. Path to Mithras

This system becomes Mithras's retrieval layer directly:
- Chunking module → manuscript preprocessing
- Dense + BM25 + RRF → shared RAG/evidence layer
- Cross-encoder reranker → evidence layer reranking stage
- Evaluation harness → reused for Mithras's own ground-truth evaluation (Section 7.3 of synopsis)

No rework needed — only swap embedding model and add multilingual support (Hindi/Marathi) when integrating.
