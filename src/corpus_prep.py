"""
corpus_prep.py — Download, clean, and trim Project Gutenberg novels.

Usage:
    python src/corpus_prep.py

Outputs cleaned text files to novels/ directory.
All three novels are already present in novels/ — this script is a no-op
if the files exist, but will re-clean them to strip Gutenberg boilerplate
and apply the word-count cap.
"""

import re
import sys
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import NOVELS_DIR, save_json, DATA_DIR

# ── Gutenberg URLs (plain text, UTF-8) ───────────────────────────────────────

GUTENBERG_URLS = {
    "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "Jane Eyre":           "https://www.gutenberg.org/files/1260/1260-0.txt",
    "Little Women":        "https://www.gutenberg.org/files/514/514-0.txt",
}

# Approx. word cap per novel — limits LLM preamble cost in Phase 3.
WORD_CAP = 30_000


# ── Cleaning helpers ─────────────────────────────────────────────────────────

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer, and prefaces."""
    start_markers = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
    ]
    end_markers = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
    ]

    for pattern in start_markers:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            text = text[m.end():]
            break

    for pattern in end_markers:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            text = text[: m.start()]
            break
            
    # Also trim prefaces, list of illustrations, etc. before the first chapter
    chapter_marker = r"^(CHAPTER\s+[IVXLCDM\d]+|Chapter\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+|[IVXLCDM]+))"
    m = re.search(chapter_marker, text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        text = text[m.start():]

    return text.strip()


def clean_text(text: str) -> str:
    """Normalize whitespace and remove illustration/figure/page tags."""
    # Remove [Illustration: ...] and [Page ...] bracketed blocks
    text = re.sub(r"\[[^\]]*\]", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining stray orphan brackets [ or ]
    text = text.replace("[", "").replace("]", "")
    # Remove italics markers
    text = text.replace("_", "")
    # Collapse horizontal whitespace (multiple spaces/tabs) on each line
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines to exactly 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace on each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def trim_to_word_cap(text: str, cap: int = WORD_CAP) -> str:
    """Trim text to approximately `cap` words at the nearest sentence boundary while preserving newlines."""
    words = text.split()
    if len(words) <= cap:
        return text
    
    current_words = 0
    cutoff_char = len(text)
    for match in re.finditer(r"\S+", text):
        current_words += 1
        if current_words == cap:
            cutoff_char = match.end()
            break

    approx = text[:cutoff_char]
    last_period = approx.rfind(".")
    if last_period != -1:
        return approx[: last_period + 1]
    return approx


# ── Download helpers ─────────────────────────────────────────────────────────

def download_novel(title: str, url: str) -> str:
    """Download raw text from Project Gutenberg. Returns raw text."""
    import urllib.request
    print(f"  Downloading {title} from {url} ...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    # Try UTF-8 first, fall back to latin-1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_corpus(word_cap: int = WORD_CAP, force_download: bool = False) -> list[dict]:
    """
    Prepare all three novels. For each:
      1. Download from Gutenberg if not already present (or force_download=True).
      2. Strip boilerplate, clean, trim.
      3. Save cleaned text back to novels/.
      4. Return list of {novel_name, text, word_count}.
    """
    corpus = []

    for title, url in GUTENBERG_URLS.items():
        filename = NOVELS_DIR / f"{title}.txt"

        if not filename.exists() or force_download:
            raw = download_novel(title, url)
            filename.write_text(raw, encoding="utf-8")
            print(f"  Saved raw text -> {filename}")
        else:
            print(f"  Found existing file: {filename}")
            raw = filename.read_text(encoding="utf-8")

        text = strip_gutenberg_boilerplate(raw)
        text = clean_text(text)
        text = trim_to_word_cap(text, cap=word_cap)

        word_count = len(text.split())
        print(f"  {title}: {word_count:,} words after cleaning + trim")

        # Overwrite the file with cleaned version
        filename.write_text(text, encoding="utf-8")

        corpus.append({"novel_name": title, "text": text, "word_count": word_count})

    # Persist corpus metadata
    meta_path = DATA_DIR / "corpus_meta.json"
    save_json(
        [{"novel_name": c["novel_name"], "word_count": c["word_count"]} for c in corpus],
        meta_path,
    )
    print(f"\nCorpus metadata saved -> {meta_path}")
    return corpus


if __name__ == "__main__":
    print("=== Corpus Preparation ===")
    NOVELS_DIR.mkdir(parents=True, exist_ok=True)
    result = prepare_corpus()
    total_words = sum(c["word_count"] for c in result)
    print(f"\nTotal corpus: {total_words:,} words across {len(result)} novels.")
