"""
build_ground_truth.py — LLM-assisted ground truth query set construction.

Workflow:
  1. Extract key passages from each novel (character intros, events, quotes).
  2. For each passage, prompt LLM to generate 3–5 query phrasings.
  3. Output draft JSON to data/eval/ground_truth_draft.json.
  4. Human reviews and adjusts grades → saves to data/eval/ground_truth.json.

Target: 40–50 total queries across 3 novels.

Usage:
    python eval/build_ground_truth.py --novel "Pride and Prejudice"
    python eval/build_ground_truth.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import NOVELS_DIR, EVAL_DIR, load_json, save_json, CHUNKS_DIR

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Seed passages per novel ───────────────────────────────────────────────────
# Each entry: {novel, passage_hint, chapter_hint}
# 'passage_hint' is a short excerpt or description for the LLM to locate.

SEED_PASSAGES = {
    "Pride and Prejudice": [
        {"hint": "Mr. Darcy's first proposal to Elizabeth Bennet", "chapter": "Chapter 34"},
        {"hint": "Elizabeth Bennet's first impression of Mr. Darcy at the Netherfield ball", "chapter": "Chapter 3"},
        {"hint": "Mr. Collins proposes to Elizabeth", "chapter": "Chapter 19"},
        {"hint": "Jane Bennet's illness at Netherfield", "chapter": "Chapter 7-8"},
        {"hint": "Lydia Bennet elopes with Wickham", "chapter": "Chapter 46"},
        {"hint": "Darcy's letter explaining his actions regarding Wickham and Jane", "chapter": "Chapter 35"},
        {"hint": "Mrs. Bennet's reaction when Bingley returns to Netherfield", "chapter": "Chapter 53"},
        {"hint": "Lady Catherine de Bourgh confronts Elizabeth about Darcy", "chapter": "Chapter 56"},
        {"hint": "Wickham's character described to Elizabeth by Colonel Fitzwilliam", "chapter": "Chapter 33"},
        {"hint": "Darcy's second proposal to Elizabeth and her acceptance", "chapter": "Chapter 58"},
    ],
    "Jane Eyre": [
        {"hint": "Jane Eyre's first encounter with Mr. Rochester on the road", "chapter": "Chapter 12"},
        {"hint": "The fire at Thornfield — Bertha Mason described", "chapter": "Chapter 27"},
        {"hint": "Rochester's interrupted wedding to Jane", "chapter": "Chapter 26"},
        {"hint": "Jane at Lowood School, Helen Burns's friendship and death", "chapter": "Chapter 8-9"},
        {"hint": "St. John Rivers proposes to Jane and asks her to go to India", "chapter": "Chapter 34"},
        {"hint": "Jane discovers Rochester's first wife is alive — Bertha Mason revealed", "chapter": "Chapter 26"},
        {"hint": "Jane returns to Thornfield to find it burned; Rochester is blind", "chapter": "Chapter 36"},
        {"hint": "Jane learns of her inheritance from her uncle in Madeira", "chapter": "Chapter 33"},
        {"hint": "Rochester disguises himself as a fortune-teller gypsy woman", "chapter": "Chapter 18-19"},
        {"hint": "Jane's departure from Lowood and arrival at Thornfield", "chapter": "Chapter 11"},
    ],
    "Little Women": [
        {"hint": "Jo March and Laurie's friendship and their first meeting", "chapter": "Chapter 5"},
        {"hint": "Beth March's illness with scarlet fever", "chapter": "Chapter 18"},
        {"hint": "Laurie proposes to Jo and she refuses him", "chapter": "Chapter 35"},
        {"hint": "Amy March burns Jo's manuscript out of anger", "chapter": "Chapter 8"},
        {"hint": "Meg March's marriage to John Brooke", "chapter": "Chapter 28"},
        {"hint": "Jo meets Professor Bhaer in New York", "chapter": "Chapter 33"},
        {"hint": "Beth March's death", "chapter": "Chapter 40"},
        {"hint": "Amy and Laurie's romance in Europe", "chapter": "Chapter 42"},
        {"hint": "The March girls and their mother Marmee discuss playing Pilgrims", "chapter": "Chapter 1"},
        {"hint": "Jo cuts and sells her hair to send money to father", "chapter": "Chapter 15"},
    ],
}

QUERIES_PER_PASSAGE = 4  # number of query phrasings to generate per passage


# ── LLM call ──────────────────────────────────────────────────────────────────

def _generate_queries_gemini(novel: str, hint: str, chapter: str) -> list[dict]:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""You are building a retrieval benchmark for the novel "{novel}".

Passage context: {hint} ({chapter})

Generate exactly {QUERIES_PER_PASSAGE} distinct query phrasings for this passage that someone might use to find it in a retrieval system. Include a mix of:
- Exact-term queries (character names, specific phrases)
- Paraphrased / semantic queries
- Relational queries (e.g. "What does X do to Y?")

For each query, also provide a relevance grade (0–3):
  3 = this passage is the primary answer
  2 = this passage is clearly relevant
  1 = marginally relevant
  0 = not relevant

Output ONLY valid JSON array, no explanation:
[
  {{"query": "...", "grade": 3}},
  ...
]"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _generate_queries_groq(novel: str, hint: str, chapter: str) -> list[dict]:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = f"""You are building a retrieval benchmark for the novel "{novel}".

Passage context: {hint} ({chapter})

Generate exactly {QUERIES_PER_PASSAGE} distinct query phrasings. Mix exact-term, paraphrased, and relational queries.
For each: grade 3=primary answer, 2=clearly relevant, 1=marginal, 0=not relevant.
Output ONLY valid JSON array: [{{"query": "...", "grade": 3}}, ...]"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    text = completion.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _get_backend():
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    raise EnvironmentError("Set GEMINI_API_KEY or GROQ_API_KEY in .env")


# ── Chunk lookup ──────────────────────────────────────────────────────────────

def _find_relevant_chunk_ids(novel: str, hint: str, strategy: str = "scene") -> list[str]:
    """
    Heuristically find chunk IDs that likely contain the passage described by hint.
    Returns top-3 most likely chunk IDs (human verifier adjusts grades).
    """
    chunk_file = CHUNKS_DIR / f"{novel.replace(' ', '_')}_{strategy}.json"
    if not chunk_file.exists():
        return []

    chunks = load_json(chunk_file)
    hint_words = set(hint.lower().split())

    scored = []
    for c in chunks:
        text_words = set(c["text"].lower().split())
        overlap = len(hint_words & text_words) / max(len(hint_words), 1)
        scored.append((overlap, c["id"]))

    scored.sort(reverse=True)
    return [cid for _, cid in scored[:3]]


# ── Main builder ──────────────────────────────────────────────────────────────

def build_ground_truth_for_novel(novel: str, backend: str) -> list[dict]:
    passages = SEED_PASSAGES.get(novel, [])
    if not passages:
        print(f"  No seed passages defined for '{novel}'")
        return []

    gt_entries = []
    for p in passages:
        hint = p["hint"]
        chapter = p["chapter"]
        print(f"  Generating queries for: {hint[:60]}...")

        try:
            if backend == "gemini":
                queries = _generate_queries_gemini(novel, hint, chapter)
            else:
                queries = _generate_queries_groq(novel, hint, chapter)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Find candidate chunk IDs (human verifies later)
        candidate_ids = _find_relevant_chunk_ids(novel, hint)

        for q in queries:
            entry = {
                "query": q.get("query", ""),
                "novel": novel,
                "passage_hint": hint,
                "chapter": chapter,
                "relevant_chunks": [
                    {"id": cid, "grade": q.get("grade", 2)}
                    for cid in candidate_ids
                ],
                "_needs_human_review": True,
            }
            gt_entries.append(entry)

    return gt_entries


def build_all(novels: list[str] | None = None) -> None:
    backend = _get_backend()
    print(f"Using LLM backend: {backend}\n")

    if novels is None:
        novels = list(SEED_PASSAGES.keys())

    all_entries = []
    for novel in novels:
        print(f"=== {novel} ===")
        entries = build_ground_truth_for_novel(novel, backend)
        all_entries.extend(entries)
        print(f"  Generated {len(entries)} entries.\n")

    draft_path = EVAL_DIR / "ground_truth_draft.json"
    save_json(all_entries, draft_path)
    print(f"Draft saved → {draft_path}")
    print(f"Total entries: {len(all_entries)}")
    print()
    print("NEXT STEP: Review ground_truth_draft.json, adjust 'relevant_chunks' IDs")
    print("           and 'grade' values, remove '_needs_human_review' flags,")
    print("           then save as data/eval/ground_truth.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ground truth query set.")
    parser.add_argument("--novel", type=str, help="Single novel to process")
    parser.add_argument("--all", action="store_true", help="Process all novels")
    args = parser.parse_args()

    if args.all:
        build_all()
    elif args.novel:
        build_all(novels=[args.novel])
    else:
        parser.print_help()
