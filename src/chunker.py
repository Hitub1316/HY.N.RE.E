"""
chunker.py — Four chunking strategies for HY.N.RE.E.

All strategies share the same interface and output the same Chunk schema:
    {id, novel, strategy, text, char_start, char_end, chapter}

Usage:
    from src.chunker import chunk

    chunks = chunk(text, strategy="scene", novel="Pride and Prejudice")
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import make_chunk, token_count, save_json, CHUNKS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_TOKENS = 256       # target chunk size in tokens
OVERLAP_RATIO = 0.20     # 20% overlap → ~51 token stride
STRIDE_TOKENS = int(CHUNK_TOKENS * (1 - OVERLAP_RATIO))  # ≈205

MIN_PARAGRAPH_TOKENS = 30   # merge paragraphs shorter than this
MAX_SCENE_CHUNK_TOKENS = 512  # hard cap before force-splitting a scene block


# ── Chapter detection ─────────────────────────────────────────────────────────

# Matches lines like: CHAPTER I / Chapter 1 / CHAPTER THE FIRST / Chapter One
_CHAPTER_RE = re.compile(
    r"^(CHAPTER\s+[IVXLCDM\d]+|Chapter\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+|[IVXLCDM]+))",
    re.IGNORECASE | re.MULTILINE,
)


def _build_chapters_info(text: str) -> list[tuple[str, int, int]]:
    """
    Return list of (chapter_label, start_char, end_char) tuples.
    Content before the first chapter header is labelled 'Preface'.
    """
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return [("Preface", 0, len(text))]

    chapters = []
    prelude = text[: matches[0].start()].strip()
    if prelude:
        chapters.append(("Preface", 0, matches[0].start()))

    for i, m in enumerate(matches):
        label = m.group(0).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append((label, start, end))

    return chapters


def _get_chapter(char_start: int, chapters_info: list[tuple[str, int, int]]) -> str:
    for label, start, end in chapters_info:
        if start <= char_start < end:
            return label
    return ""


# ── Strategy: fixed token sliding window ─────────────────────────────────────

def _chunk_fixed(text: str, novel: str) -> list[dict]:
    """256-token sliding window with 20% overlap."""
    from utils import get_tokenizer
    tokenizer = get_tokenizer()
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    chapters_info = _build_chapters_info(text)

    chunks = []
    idx = 0
    chunk_num = 0
    while idx < len(offsets):
        window_offsets = offsets[idx: idx + CHUNK_TOKENS]
        if not window_offsets:
            break
        char_start = window_offsets[0][0]
        char_end = window_offsets[-1][1]
        chunk_text = text[char_start:char_end]
        chapter = _get_chapter(char_start, chapters_info)

        chunks.append(make_chunk(
            chunk_id=f"{novel}__fixed__{chunk_num:04d}",
            novel=novel,
            strategy="fixed",
            text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            chapter=chapter,
        ))
        idx += STRIDE_TOKENS
        chunk_num += 1
        if idx >= len(offsets):
            break

    return chunks


# ── Strategy: paragraph ───────────────────────────────────────────────────────

def _chunk_paragraph(text: str, novel: str) -> list[dict]:
    """Split on blank lines, merge short paragraphs up to CHUNK_TOKENS budget."""
    raw_paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chapters_info = _build_chapters_info(text)

    chunks = []
    buffer = []
    buffer_tokens = 0
    char_cursor = 0
    chunk_num = 0

    def flush():
        nonlocal buffer, buffer_tokens, chunk_num, char_cursor
        if not buffer:
            return
        merged = "\n\n".join(buffer)
        
        search_text = merged[:50] if len(merged) > 50 else merged
        char_start = text.find(search_text, char_cursor)
        if char_start == -1:
            char_start = text.find(search_text)
            
        char_end = -1
        chapter = ""
        if char_start != -1:
            char_end = char_start + len(merged)
            char_cursor = char_end
            chapter = _get_chapter(char_start, chapters_info)

        chunks.append(make_chunk(
            chunk_id=f"{novel}__paragraph__{chunk_num:04d}",
            novel=novel,
            strategy="paragraph",
            text=merged,
            char_start=char_start,
            char_end=char_end,
            chapter=chapter,
        ))
        buffer = []
        buffer_tokens = 0
        chunk_num += 1

    for para in raw_paragraphs:
        para_tokens = token_count(para)
        if buffer_tokens + para_tokens > CHUNK_TOKENS:
            flush()
        buffer.append(para)
        buffer_tokens += para_tokens

    flush()
    return chunks


# ── Strategy: scene-aware ─────────────────────────────────────────────────────

def _is_scene_break(prev_line: str, curr_line: str) -> bool:
    """
    Heuristic scene shift detection within a chapter:
      - curr_line starts a new sentence (capital letter after blank) AND
        prev_line ends with sentence-terminating punctuation.
    """
    if not prev_line or not curr_line:
        return False
    ends_sentence = prev_line.rstrip().endswith((".", "!", "?", '"', "'"))
    starts_new = bool(curr_line) and curr_line[0].isupper()
    return ends_sentence and starts_new


def _split_scene_blocks(chapter_text: str) -> list[str]:
    """
    Within a chapter, group paragraphs into scene-level blocks targeting CHUNK_TOKENS (~256 tokens).
    Never breaks mid-paragraph unless a single paragraph exceeds MAX_SCENE_CHUNK_TOKENS.
    """
    paras = [p.strip() for p in re.split(r"\n\n+", chapter_text) if p.strip()]
    if not paras:
        return []

    blocks = []
    current_block = [paras[0]]
    current_tokens = token_count(paras[0])

    for i in range(1, len(paras)):
        curr = paras[i]
        curr_tokens = token_count(curr)

        if current_tokens + curr_tokens > CHUNK_TOKENS:
            blocks.append("\n\n".join(current_block))
            current_block = [curr]
            current_tokens = curr_tokens
        else:
            current_block.append(curr)
            current_tokens += curr_tokens

    if current_block:
        blocks.append("\n\n".join(current_block))

    return blocks


def _subdivide_large_block(block: str, novel: str, strategy: str, chapter: str, base_id: str, original_text: str, cursor_start: int) -> list[dict]:
    """Force-split a scene block that exceeds MAX_SCENE_CHUNK_TOKENS."""
    from utils import get_tokenizer
    tokenizer = get_tokenizer()
    encoded = tokenizer(block, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]

    sub_chunks = []
    idx = 0
    sub_num = 0
    while idx < len(offsets):
        window_offsets = offsets[idx: idx + CHUNK_TOKENS]
        if not window_offsets:
            break
        rel_start = window_offsets[0][0]
        rel_end = window_offsets[-1][1]
        chunk_text = block[rel_start:rel_end]

        char_start = cursor_start + rel_start if cursor_start != -1 else -1
        char_end = cursor_start + rel_end if cursor_start != -1 else -1

        sub_chunks.append(make_chunk(
            chunk_id=f"{base_id}_sub{sub_num:02d}",
            novel=novel,
            strategy=strategy,
            text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            chapter=chapter,
        ))
        idx += STRIDE_TOKENS
        sub_num += 1

    return sub_chunks


def _chunk_scene(text: str, novel: str, strategy: str = "scene") -> list[dict]:
    """
    Split by chapter → within each chapter split by scene-shift heuristics.
    Large scene blocks are force-subdivided at CHUNK_TOKENS.
    """
    chapters_info = _build_chapters_info(text)
    chunks = []
    chunk_num = 0
    char_cursor = 0

    for chapter_label, start_idx, end_idx in chapters_info:
        chapter_text = text[start_idx:end_idx].strip()
        blocks = _split_scene_blocks(chapter_text)
        for block in blocks:
            block_tokens = token_count(block)
            base_id = f"{novel}__{strategy}__{chunk_num:04d}"
            
            search_text = block[:50] if len(block) > 50 else block
            char_start = text.find(search_text, char_cursor)
            if char_start == -1:
                char_start = text.find(search_text)
            
            char_end = -1
            if char_start != -1:
                char_end = char_start + len(block)

            if block_tokens > MAX_SCENE_CHUNK_TOKENS:
                sub = _subdivide_large_block(block, novel, strategy, chapter_label, base_id, text, char_start if char_start != -1 else char_cursor)
                chunks.extend(sub)
                if char_end != -1:
                    char_cursor = char_end
            else:
                if char_end != -1:
                    char_cursor = char_end
                chunks.append(make_chunk(
                    chunk_id=base_id,
                    novel=novel,
                    strategy=strategy,
                    text=block,
                    char_start=char_start,
                    char_end=char_end,
                    chapter=chapter_label,
                ))
            chunk_num += 1

    return chunks


# ── Strategy: contextual ──────────────────────────────────────────────────────

def _chunk_contextual(text: str, novel: str) -> list[dict]:
    """
    Same as scene-aware, but each chunk gets an LLM preamble prepended.
    Preamble generation is deferred to contextual_enrichment.py — this
    function returns scene chunks tagged with strategy='contextual'.
    The enrichment step later mutates chunk['text'] in place.
    """
    chunks = _chunk_scene(text, novel, strategy="contextual")
    return chunks


# ── Public interface ──────────────────────────────────────────────────────────

STRATEGIES = ("fixed", "paragraph", "scene", "contextual")


def chunk(text: str, strategy: str, novel: str, **kwargs) -> list[dict]:
    """
    Chunk `text` using the given `strategy`.

    Args:
        text:     Cleaned novel text.
        strategy: One of 'fixed', 'paragraph', 'scene', 'contextual'.
        novel:    Novel title (used in chunk IDs).

    Returns:
        List of Chunk dicts conforming to the canonical schema.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")

    if strategy == "fixed":
        return _chunk_fixed(text, novel)
    elif strategy == "paragraph":
        return _chunk_paragraph(text, novel)
    elif strategy == "scene":
        return _chunk_scene(text, novel)
    elif strategy == "contextual":
        return _chunk_contextual(text, novel)


def chunk_and_save(text: str, strategy: str, novel: str) -> list[dict]:
    """Chunk and persist to data/chunks/{novel}_{strategy}.json."""
    chunks = chunk(text, strategy, novel)
    out_path = CHUNKS_DIR / f"{novel.replace(' ', '_')}_{strategy}.json"
    save_json(chunks, out_path)
    print(f"  [{strategy}] {len(chunks)} chunks -> {out_path}")
    return chunks


if __name__ == "__main__":
    # Quick smoke test: chunk Pride and Prejudice with all strategies
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import NOVELS_DIR

    novel = "Pride and Prejudice"
    path = NOVELS_DIR / f"{novel}.txt"
    if not path.exists():
        print(f"ERROR: {path} not found. Run corpus_prep.py first.")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    print(f"=== Chunking smoke test: {novel} ({len(text.split()):,} words) ===\n")

    for strat in ("fixed", "paragraph", "scene"):  # skip contextual (needs LLM)
        chunks = chunk(text, strat, novel)
        token_lengths = [token_count(c["text"]) for c in chunks]
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        print(f"  {strat:12s}: {len(chunks):4d} chunks | avg {avg_tokens:.0f} tokens")
