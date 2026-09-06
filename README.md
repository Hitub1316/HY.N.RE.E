# HY.N.RE.E — Hybrid Narrative Retrieval Engine

A research-grade hybrid information-retrieval system evaluated over a public-domain fiction corpus.
Proves which retrieval strategy (lexical, dense, hybrid, reranked, contextual) performs best on narrative text.
Designed to be lifted directly into [Mithras](https://github.com/Hitub1316) as its retrieval layer.

---

## Corpus

| Novel | Author | Source |
|---|---|---|
| Pride and Prejudice | Jane Austen | Project Gutenberg #1342 |
| Jane Eyre | Charlotte Brontë | Project Gutenberg #1260 |
| Little Women | Louisa May Alcott | Project Gutenberg #514 |

Each novel trimmed to ~30k words to cap LLM contextual-preamble generation cost.

---

## Architecture

```
Novel corpus
     ↓
Structure-aware chunking (scene/paragraph level)
     ↓
Contextual enrichment (LLM preamble, prepended, cached)
     ↓
     ┌──────────────┬──────────────┐
     ↓              ↓
Dense retrieval   BM25 retrieval
(all-MiniLM-L6-v2 + Qdrant)   (rank_bm25)
     ↓              ↓
     └──────┬───────┘
            ↓
      RRF Fusion (k=60)
            ↓
    Cross-Encoder Reranker
    (ms-marco-MiniLM-L-6-v2)
            ↓
      Top-K evidence passages
            ↓
       Evaluation harness
```

---

## Setup

### 1. API Key (for contextual preamble generation)

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
cp .env.example .env
# Edit .env and paste your key:
# GEMINI_API_KEY=your_key_here
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Corpus files

Corpus files are in `novels/`. If any are missing, `corpus_prep.py` will download them:

```bash
python src/corpus_prep.py
```

---

## Running the Pipeline

### Phase 1: Chunk + Index

```bash
# Chunk all novels with all strategies (or pick one)
python src/chunker.py                       # smoke test on P&P

# Build Qdrant + BM25 indices
python src/indexer.py --strategy scene      # or: --strategy all
```

### Phase 2: Evaluate

```bash
# Build ground truth (LLM-assisted, then hand-verify)
python eval/build_ground_truth.py --all
# → review data/eval/ground_truth_draft.json
# → save verified version as data/eval/ground_truth.json
```

### Phase 3: Full demo + ablation

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Results

> _To be filled after Weekend 2, Day 2 evaluation run._

### System Ablation

| System | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|
| BM25 only | — | — | — |
| Dense only | — | — | — |
| Hybrid (RRF) | — | — | — |
| Hybrid + Rerank | — | — | — |
| Contextual Hybrid + Rerank | — | — | — |

### Chunking Ablation (Hybrid+Rerank, fixed)

| Chunking Strategy | Recall@5 | MRR |
|---|---|---|
| Fixed token | — | — |
| Paragraph | — | — |
| Scene-aware | — | — |
| Contextual scene-aware | — | — |

---

## Mithras Reusability Map

| HY.N.RE.E Module | Mithras Component |
|---|---|
| `src/corpus_prep.py` | Manuscript preprocessing |
| `src/chunker.py` (scene strategy) | Document chunking layer |
| `src/contextual_enrichment.py` | Context-aware chunk enrichment |
| `src/indexer.py` | Dense + lexical index builder |
| `src/retriever.py` | RAG/evidence retrieval layer |
| `src/evaluator.py` | Ground-truth evaluation harness (Section 7.3) |

No rework needed for integration — only swap the embedding model and add multilingual support (Hindi/Marathi) when integrating.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Qdrant (local path mode) |
| Lexical retrieval | `rank-bm25` |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Contextual preambles | Gemini Flash (primary) / Groq (fallback), cached |
| Demo | Jupyter notebook |
| Evaluation | Custom harness (P@K, R@K, MRR, NDCG@10) |
