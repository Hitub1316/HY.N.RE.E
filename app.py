"""
app.py — Standalone Streamlit UI for HY.N.RE.E (Hybrid Narrative Retrieval Engine).

Usage:
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from retriever import HybridRetriever
from utils import load_json, EVAL_DIR, token_count
from parsers import parse_file
from chunker import chunk
from contextual_enrichment import enrich_chunks
from vector_store import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import copy

st.set_page_config(
    page_title="HY.N.RE.E — Narrative Retrieval Engine",
    page_icon="📖",
    layout="wide"
)

# Initialize session state for custom uploaded novels
if "custom_novels" not in st.session_state:
    st.session_state["custom_novels"] = {}  # title -> {strategy -> list[dict]}
if "custom_bm25" not in st.session_state:
    st.session_state["custom_bm25"] = {}    # title -> {strategy -> BM25Okapi}

# ── Title Header ──────────────────────────────────────────────────────────────
st.title("📖 HY.N.RE.E — Hybrid Narrative Retrieval Engine")
st.markdown(
    "A research-grade hybrid retrieval engine evaluated across *Pride and Prejudice*, *Jane Eyre*, and *Little Women*. "
    "Supports live narrative passage search, full ablation metrics, and custom novel upload (.txt, .pdf, .docx)."
)

# ── System Configuration Mapping ──────────────────────────────────────────────
SYSTEM_CONFIGS = {
    "BM25 only": {"strategy": "scene", "method": "bm25"},
    "Dense only": {"strategy": "scene", "method": "dense"},
    "Hybrid (RRF)": {"strategy": "scene", "method": "hybrid"},
    "Hybrid + Rerank": {"strategy": "scene", "method": "hybrid_reranked"},
    "Contextual Hybrid + Rerank": {"strategy": "contextual", "method": "hybrid_reranked"},
}

BASE_NOVEL_OPTIONS = ["All Novels", "Pride and Prejudice", "Jane Eyre", "Little Women"]
all_novel_options = BASE_NOVEL_OPTIONS + list(st.session_state["custom_novels"].keys())

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Retrieval Settings")

selected_system = st.sidebar.selectbox("Retrieval System Config", list(SYSTEM_CONFIGS.keys()), index=3)
selected_novel = st.sidebar.selectbox("Novel Filter", all_novel_options, index=0)
top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=5)
expand_neighbors_toggle = st.sidebar.checkbox("Expand Neighbor Chunks (Evidence Window)", value=False)

st.sidebar.markdown("---")

# ── Custom Novel Upload Sidebar Section ───────────────────────────────────────
st.sidebar.header("📚 Upload Custom Novel")
st.sidebar.info("ℹ️ **Note:** Uploaded custom novels and their indices are maintained in-session only and will be cleared when the application restarts.")

uploaded_file = st.sidebar.file_uploader(
    "Upload a novel file (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    help="Upload a book or long document up to 10MB."
)

enable_enrichment = st.sidebar.checkbox(
    "Enable Contextual Enrichment (LLM preambles)",
    value=False,
    help="Prepend narrative preambles to chunks before embedding. Uses cached preambles or rule-based synthesis if API key is unconfigured."
)

if uploaded_file is not None:
    # 10MB size limit check
    if uploaded_file.size > 10 * 1024 * 1024:
        st.sidebar.error("File size exceeds 10MB limit.")
    else:
        custom_novel_title = Path(uploaded_file.name).stem.replace("_", " ").title()
        
        if st.sidebar.button(f"Index '{custom_novel_title}'", type="primary"):
            with st.spinner(f"Extracting and indexing '{custom_novel_title}'..."):
                file_bytes = uploaded_file.read()
                try:
                    text = parse_file(uploaded_file.name, file_bytes)
                except Exception as e:
                    st.sidebar.error(f"Failed to parse file: {e}")
                    text = ""

                words = text.split()
                if len(words) < 500:
                    st.sidebar.warning(f"Text is too short ({len(words)} words). Minimum length is 500 words.")
                else:
                    if len(words) > 50000 and enable_enrichment:
                        st.sidebar.warning(f"Large text ({len(words):,} words). Contextual enrichment may take time or fall back to rule-based preambles if LLM limits hit.")
                    
                    # 1. Chunk document using scene strategy
                    progress_bar = st.progress(0.2, text="Chunking novel into scene blocks...")
                    scene_chunks = chunk(text, strategy="scene", novel=custom_novel_title)

                    # 2. Contextual enrichment if opted-in
                    if enable_enrichment:
                        progress_bar.progress(0.5, text="Enriching chunks with narrative preambles...")
                        contextual_chunks = enrich_chunks(copy.deepcopy(scene_chunks), custom_novel_title)
                    else:
                        contextual_chunks = scene_chunks

                    # 3. Build dense vector embeddings & Qdrant collection for custom novel
                    progress_bar.progress(0.7, text="Building dense embeddings and Qdrant index...")
                    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                    store = QdrantVectorStore()
                    custom_coll = f"hynree_custom_{custom_novel_title.replace(' ', '_').lower()}"
                    
                    if store.collection_exists(custom_coll):
                        store.delete_collection(custom_coll)
                    
                    store.create_collection(custom_coll, vector_size=384, distance="Cosine")
                    
                    embed_texts = [c.get("contextual_text") or c["text"] for c in contextual_chunks]
                    embeddings = embed_model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True)
                    
                    points = []
                    for j, (chunk_obj, emb) in enumerate(zip(contextual_chunks, embeddings)):
                        points.append({
                            "id": j,
                            "vector": emb.tolist(),
                            "payload": {
                                "chunk_id": chunk_obj["id"],
                                "novel": custom_novel_title,
                                "strategy": "contextual" if enable_enrichment else "scene",
                                "chapter": chunk_obj.get("chapter", ""),
                                "text": chunk_obj["text"],
                                "contextual_text": chunk_obj.get("contextual_text", ""),
                                "char_start": chunk_obj.get("char_start", -1),
                                "char_end": chunk_obj.get("char_end", -1),
                            },
                        })
                    store.upsert(collection_name=custom_coll, points=points)

                    # 4. Build BM25 index for custom novel
                    progress_bar.progress(0.9, text="Building lexical BM25 index...")
                    tokenized = [c["text"].lower().split() for c in contextual_chunks]
                    bm25_obj = BM25Okapi(tokenized)

                    # Save in session state
                    st.session_state["custom_novels"][custom_novel_title] = {
                        "scene": scene_chunks,
                        "contextual": contextual_chunks,
                    }
                    st.session_state["custom_bm25"][custom_novel_title] = bm25_obj

                    progress_bar.progress(1.0, text="Indexing complete!")
                    st.sidebar.success(f"Successfully indexed '{custom_novel_title}' ({len(contextual_chunks)} chunks)!")
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("HY.N.RE.E Retrieval Pipeline | Powered by Qdrant, BM25 & MS-MARCO Reranker")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_search, tab_ablation, tab_about = st.tabs(["🔍 Live Search", "📊 Ablation Benchmark", "ℹ️ About & Methodology"])

# ── Tab 1: Live Search ────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Interactive Narrative Passage Search")
    
    default_query = "What does Elizabeth Bennet think of Mr. Darcy at the Netherfield ball?"
    query = st.text_input("Enter search query:", value=default_query)
    
    if st.button("Search Passages", type="primary") or query:
        cfg = SYSTEM_CONFIGS[selected_system]
        
        # Check if searching a custom in-session uploaded novel
        if selected_novel in st.session_state["custom_novels"]:
            with st.spinner(f"Searching custom novel '{selected_novel}'..."):
                custom_chunks = st.session_state["custom_novels"][selected_novel]["contextual" if cfg["strategy"] == "contextual" else "scene"]
                bm25_obj = st.session_state["custom_bm25"][selected_novel]
                
                # Search BM25 or Dense over custom novel
                query_tokens = query.lower().split()
                bm25_scores = bm25_obj.get_scores(query_tokens)
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
                
                filtered_results = []
                for rank, idx in enumerate(top_indices, start=1):
                    c = custom_chunks[idx]
                    filtered_results.append({
                        "rank": rank,
                        "chunk_id": c["id"],
                        "novel": c["novel"],
                        "chapter": c.get("chapter", ""),
                        "score": float(bm25_scores[idx]),
                        "text": c["text"],
                        "retrieval_method": "custom_bm25",
                    })
        else:
            with st.spinner(f"Running {selected_system} search..."):
                retriever = HybridRetriever(strategy=cfg["strategy"])
                raw_results = retriever.retrieve(
                    query=query,
                    method=cfg["method"],
                    top_k=top_k * 3 if selected_novel != "All Novels" else top_k,
                    expand_neighbors=expand_neighbors_toggle,
                )
                
                # Filter by novel if selected
                filtered_results = []
                for r in raw_results:
                    target_item = r["target"] if expand_neighbors_toggle else r
                    if selected_novel == "All Novels" or target_item.get("novel") == selected_novel:
                        filtered_results.append(r)
                    if len(filtered_results) >= top_k:
                        break

        st.success(f"Found {len(filtered_results)} passages using **{selected_system}**")
        st.markdown("---")

        for idx, item in enumerate(filtered_results, start=1):
            if expand_neighbors_toggle and isinstance(item, dict) and "target" in item:
                target = item["target"]
                before = item["before"]
                after = item["after"]
                
                st.markdown(f"### Result {idx}: {target['novel']} — {target['chapter']}")
                st.caption(f"**Chunk ID**: `{target['chunk_id']}` | **Score**: `{target['score']:.4f}` | **Method**: `{target['retrieval_method']}`")
                
                col_b, col_t, col_a = st.columns(3)
                
                with col_b:
                    st.markdown("**Preceding Chunk (Before)**")
                    if before:
                        st.info(before["text"])
                    else:
                        st.warning("*(Chapter Boundary — First Chunk)*")
                
                with col_t:
                    st.markdown("**Matched Target Chunk**")
                    st.success(target["text"])

                with col_a:
                    st.markdown("**Following Chunk (After)**")
                    if after:
                        st.info(after["text"])
                    else:
                        st.warning("*(Chapter Boundary — Last Chunk)*")

                st.markdown("---")
            else:
                r = item
                st.markdown(f"### #{r.get('rank', idx)} {r['novel']} — {r.get('chapter', '')}")
                st.caption(f"**Chunk ID**: `{r['chunk_id']}` | **Score**: `{r['score']:.4f}` | **Method**: `{r.get('retrieval_method', selected_system)}`")
                st.markdown(f"> {r['text']}")
                st.markdown("---")

# ── Tab 2: Ablation Benchmark ─────────────────────────────────────────────────
with tab_ablation:
    st.subheader("Precomputed 5-System Ablation Results (49 Queries)")
    
    ablation_file = EVAL_DIR / "ablation_results.json"
    if ablation_file.exists():
        data = load_json(ablation_file)
        df_overall = pd.DataFrame(data["overall_summary"]).T
        df_by_category = pd.DataFrame(data["by_category_summary"]).T
        
        st.markdown("### 1. Overall System Metrics")
        st.dataframe(df_overall, use_container_width=True)
        
        st.markdown("### 2. Metric Comparison Bar Chart")
        st.bar_chart(df_overall[["Recall@5", "MRR", "NDCG@10"]])
        
        st.markdown("### 3. Per-Category Breakdown (Recall@5 & MRR)")
        st.dataframe(df_by_category, use_container_width=True)
    else:
        st.warning("Ablation results file not found. Run `python src/evaluator.py` first.")

# ── Tab 3: About & Methodology ────────────────────────────────────────────────
with tab_about:
    st.subheader("About HY.N.RE.E")
    st.markdown(
        """
## 1. Project at a Glance

HY.N.RE.E is a search engine for novels. It finds the exact passages that answer a question about a story.

**Example question:**

> "What does Elizabeth know about Darcy's proposal before she visits Pemberley?"

The system does not just chat. It retrieves evidence. Then it measures how good that retrieval is. That measurement is the core contribution.

| Item                | Detail                                                                 |
| ------------------- | ---------------------------------------------------------------------- |
| **Type**            | Information-retrieval research project                                 |
| **Corpus**          | Three public-domain novels from Project Gutenberg, about 492,000 words |
| **Chunks**          | 3,098 scene-level chunks                                               |
| **Test queries**    | 49 hand-built queries across 7 categories                              |
| **Role in Mithras** | Becomes the retrieval layer of the final year project                  |

---

## 2. Why This Project

**Problem.** Most student RAG projects are "chat with PDF". They use one search method and never test it. They cannot say whether the method works.

**Two weaknesses of single methods:**

* Dense (meaning-based) search misses exact names and phrases.
* Keyword search misses paraphrased questions.

**Our answer.** Combine both. Re-rank the results. Add context to each chunk. Then prove with numbers which setup wins.

---

## 3. Key Concepts in Plain English

**RAG (Retrieval-Augmented Generation).**
A language model answers using passages fetched from a document collection. Retrieval quality decides answer quality. If retrieval fails, the answer fails.

**Chunking.**
Splitting a long novel into small pieces. A search engine cannot index a whole novel as one unit. We split by scene, not fixed length, so each chunk keeps one coherent event.

**Embedding.**
A list of numbers that captures the meaning of text. Similar meanings give similar numbers. We use a local sentence-transformers model.

**Dense retrieval.**
Embed the query. Find chunks with the closest embeddings. Good at meaning. Weak at exact words. We store vectors in Qdrant, a vector database.

**BM25 (sparse / keyword retrieval).**
Classic keyword scoring. Rewards rare matching words and penalizes long documents. Good for names like "Pemberley". Blind to synonyms.

**Hybrid search.**
Run dense and BM25 together. Merge the two ranked lists.

**RRF (Reciprocal Rank Fusion).**
The merging formula. Each chunk scores `1 / (k + rank)` in each list, with `k = 60`. Scores are summed. A chunk ranked high in both lists wins. It needs no score calibration between methods.

**Cross-encoder re-ranker.**
A model that reads the query and a chunk together and scores relevance. More accurate than embeddings. Also slower. So it only re-ranks the top fused candidates.

**Contextual retrieval.**
A chunk like "She refused and left the room" is ambiguous alone. An LLM writes a short preamble stating who, where, and what chapter. The preamble is attached before indexing. Anthropic reported this technique cut retrieval failures substantially.

**Ablation.**
Test the system with one component switched on at a time. This shows what each part contributes.

---

## 4. System Architecture

```mermaid
flowchart TD
    A[Novel text<br/>3 books] --> B[Scene-aware chunking<br/>3,098 chunks]
    B --> C[Contextual enrichment<br/>LLM writes a preamble per chunk, cached]

    C --> D[Dense retrieval<br/>Embeddings + Qdrant]
    C --> E[BM25 retrieval<br/>Keyword index]

    D --> F[RRF fusion<br/>k = 60]
    E --> F

    F --> G[Cross-encoder<br/>re-ranking]
    G --> H[Top-K evidence passages]
    H --> I[Evaluation harness<br/>Metrics]
```

---

## 5. Components and Status

| Component                           | Technology                       | Status                     |
| ----------------------------------- | -------------------------------- | -------------------------- |
| **Corpus preparation**              | Project Gutenberg text files     | Done                       |
| **Scene-aware chunking**            | Chapter and paragraph heuristics | Done                       |
| **Dense retrieval**                 | sentence-transformers + Qdrant   | Done                       |
| **BM25 retrieval**                  | rank_bm25                        | Done                       |
| **RRF fusion**                      | Custom implementation, `k = 60`  | Done                       |
| **Cross-encoder re-ranker**         | MiniLM cross-encoder             | Done                       |
| **Contextual enrichment**           | Gemini and Groq APIs, cached     | Done (rebuilt after fixes) |
| **Evaluation harness**              | Custom metrics code, 49 queries  | Done                       |
| **Final ablation rerun**            | Evaluator on rebuilt indices     | In progress                |
| **README and methodology write-up** | Markdown                         | In progress                |

---

## 6. How We Measure Quality

We built **49 ground-truth queries**. They cover 7 categories:

* Exact entity
* Semantic
* Relationship
* Temporal
* Event
* Canon rule
* Paraphrase

Each query has known relevant passages.

| Metric          | Meaning                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| **Recall@5**    | Share of relevant passages found in the top 5 results                                             |
| **Precision@5** | Share of the top 5 results that are relevant                                                      |
| **MRR**         | Average of `1 / rank` of the first relevant result. Higher means the right answer appears earlier |
| **NDCG@10**     | Rewards putting highly relevant passages near the top of the top 10                               |

Five systems are compared:

1. BM25 only
2. Dense only
3. Hybrid (RRF)
4. Hybrid + Rerank
5. Contextual Hybrid + Rerank

---

## 7. Results So Far (Preliminary)

First full ablation on the 492k-word corpus:

| System                         |   Recall@5 |        MRR |    NDCG@10 |
| ------------------------------ | ---------: | ---------: | ---------: |
| **BM25 only**                  |     0.2449 |     0.2389 |     0.2029 |
| **Dense only**                 |     0.1633 |     0.2536 |     0.1739 |
| **Hybrid (RRF)**               |     0.2347 |     0.2678 |     0.2144 |
| **Hybrid + Rerank**            | **0.2449** | **0.3027** | **0.2447** |
| **Contextual Hybrid + Rerank** |     0.2347 |     0.2929 |     0.2333 |

### What this shows

* **Hybrid + Rerank** gave the best MRR and NDCG. Fusion plus re-ranking beats either single method.
* **BM25** won on exact-entity queries. **Dense** won on paraphrase-style queries. This supports using both.
* **Dense alone** had the lowest Recall@5.

---

## 8. Problems Faced and How We Solved Them

| Problem                         | Cause                                        | Fix                                                |
| ------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| 82% of preambles were templates | API keys were not visible to the subprocess  | Set environment variables and regenerate preambles |
| Gemini API returned 404         | `gemini-1.5-flash` was retired               | Switched to a current Gemini model                 |
| Cache write errors              | Cache file saved on every thread completion  | Batched cache writes                               |
| Qdrant lock conflict            | Stale lock from an interrupted run           | Cleared storage and rebuilt indices                |
| Unreliable results table        | Report was posted before evaluator succeeded | Re-ran evaluation and verified raw JSON output     |

---

## 9. Connection to Mithras

Mithras is an analysis-first narrative intelligence layer for fiction writers. It must find evidence in a manuscript to check continuity, voice, and character facts.

HY.N.RE.E becomes its **evidence layer**:

* **Chunking module** becomes manuscript preprocessing.
* **Hybrid retrieval and reranking** become the shared evidence engine used by the agents.
* **Evaluation harness** is reused to test Mithras itself.
* Later change: swap the embedding model and add **Hindi and Marathi support**.

"""
    )
