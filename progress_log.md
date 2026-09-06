# Comprehensive HY.N.RE.E Project Progress Log

## 1. Setup
- **Goal**: 4 chunking strategies (`fixed`, `paragraph`, `scene`, `contextual`) across 3 novels (*Pride and Prejudice*, *Jane Eyre*, *Little Women*).
- **Schema**: `{id, novel, strategy, text, char_start, char_end, chapter}`.

---

## 2. First Bug Set
### Reported Bugs:
- `char_start`/`char_end` = `-1` on every chunk.
- `chapter` field empty on every chunk.
- Content wrong — chunks contained Saintsbury's critical preface, not the novel opening.

### Root Cause:
- **`char_start`/`char_end` bug**: Chunking code decoded token IDs via HuggingFace `AutoTokenizer.decode()`. This lowercased characters and altered spacing around punctuation, so subsequent `.find()` lookups against the original (unlowercased) text failed and returned `-1`.
- **`chapter` bug**: Downstream of the position tracking failure — position tracking failures prevented chapter-boundary mapping from working.
- **Content bug**: Boilerplate stripping matched the Gutenberg `*** START OF ***` marker but did not strip the editor's preface/critical essay preceding Chapter I.

### Fix:
- Replaced `.find()`-based position tracking with `return_offsets_mapping=True` on the tokenizer, extracting exact character offsets directly from token spans.
- Wired `_build_chapters_info` to map chapter labels to character-position ranges.
- Added header scanning (`CHAPTER\s+[IVXLCDM\d]+`) to `corpus_prep.py` to slice text starting cleanly at Chapter I, past the preface.

---

## 3. Stray Bracket Artifact
### Reported Bug:
- Chunk text contained `"Chapter I.] It is a truth..."` — stray `]` character.

### Diagnostic Step:
- Direct read of first 200 chars of cleaned `.txt` file, bypassing the chunker.

### Diagnostic Context:
- An earlier version of this diagnostic script (reading `corpus_meta.json` as a dict and indexing by novel name) failed with:
  `TypeError: list indices must be integers or slices, not str`
  because `corpus_meta.json` was actually structured as a list of dicts, not a top-level dict. This was a scratch-script bug in the diagnostic script itself — not a defect in the pipeline code (`corpus_prep.py`, `chunker.py`). Once the script was corrected to read the `.txt` file directly, it ran cleanly and confirmed the source text had no bracket at position 0.

### Root Cause of the Actual `]` Bug:
- Gutenberg illustration/page tags (`[Illustration...]`, `[Page ...]`) left orphan `]` characters when only partially stripped by the original bracket-removal regex, which targeted `[Illustration...]` specifically but missed other bracketed forms.

### Fix:
- Updated `clean_text()` to strip all bracketed blocks (`\[[^\]]*\]`) and purge any remaining orphan `[`/`]` characters.

---

## 4. Whitespace Artifact
### Reported Bug:
- Removing brackets left double spaces (e.g., `"hearing it.”  This was"`).

### Fix:
- Added horizontal whitespace collapsing to `clean_text()`: `re.sub(r"[ \t]+", " ", text)`, plus line-stripping.

### Verification:
- Spot-check on `fixed`, `paragraph`, and `scene` strategies for *Pride and Prejudice*. All 3 passed: `chapter` field populated, no whitespace artifacts, `char_start`/`char_end` correctly tracked (e.g., `0..1060`, `840..1837`).

---

## 5. Cross-Novel Strategy Inconsistency
### Reported Bug:
- `scene` strategy produced ~1060-char chunks for *Pride and Prejudice* but ~350-char chunks (single paragraphs) for *Jane Eyre* and *Little Women*.

### Root Cause:
- `_is_scene_break` combined with `MIN_PARAGRAPH_TOKENS = 30` triggered on nearly every paragraph boundary in *Jane Eyre* and *Little Women*, flushing each paragraph as its own scene block. *Pride and Prejudice*'s opening dialogue-heavy paragraphs happened to avoid early breaks, producing one large block instead.

### Fix:
- `_split_scene_blocks()` refactored to accumulate paragraphs up to `CHUNK_TOKENS` (~256 tokens) per scene block, consistently across all 3 novels.

### Follow-up Check:
- After the fix, `scene` and `paragraph` strategies produced identical opening chunk lengths (`[823, 957, 944]`) across all 3 novels — flagged as suspicious, tested directly rather than assumed correct.

### Empirical Divergence Test:

| Novel | Paragraph strategy: chapter-crossing chunks | Scene strategy: chapter-crossing chunks |
| :--- | :---: | :---: |
| **Pride and Prejudice** | 16 | 0 |
| **Jane Eyre** | 7 | 0 |
| **Little Women** | 7 | 0 |

- Confirmed via direct text inspection: `P&P__paragraph__0004` ends with `"...CHAPTER II."` appended mid-chunk. `P&P__scene__0004` hard-flushes at char 4487 (end of Chapter I); next scene chunk starts cleanly at char 4489 with `"CHAPTER II."`.

### Conclusion:
- Paragraph and Scene strategies are genuinely distinct. Paragraph is chapter-blind; Scene strictly respects chapter boundaries. Not a duplicate strategy.

---

## 6. Current State — Persisted Datasets

All 9 non-LLM chunk datasets (3 novels × 3 strategies) generated and saved to `data/chunks/`:

| Novel | Strategy | Chunk Count |
| :--- | :--- | :---: |
| **Pride and Prejudice** | Fixed | 191 |
| **Pride and Prejudice** | Paragraph | 184 |
| **Pride and Prejudice** | Scene | 188 |
| **Jane Eyre** | Fixed | 193 |
| **Jane Eyre** | Paragraph | 186 |
| **Jane Eyre** | Scene | 190 |
| **Little Women** | Fixed | 193 |
| **Little Women** | Paragraph | 187 |
| **Little Women** | Scene | 188 |

---

## Next Steps
- **Step 4**: Contextual enrichment (LLM preambles), tested on a small batch (5–10 chunks) before scaling to full corpus.
- Ground-truth query set generation and evaluation harness run.
