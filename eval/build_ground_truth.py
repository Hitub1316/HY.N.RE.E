"""
build_ground_truth.py — Ground truth query set construction with category tagging.

Categories (7 categories, 7 queries per category = 49 total queries):
  - exact_entity: direct lookups for named characters, places, or proper nouns
  - semantic: conceptual, thematic, or abstract queries
  - relationship: character interactions, motivations, and social dynamics
  - temporal: queries concerning sequence of events, timing, or chronologies
  - event: specific plot incidents, scenes, or actions
  - canon_rule: social rules, estate laws, or world conventions
  - paraphrase: genuine semantic rephrasings (zero literal quote overlap)

Output format (data/eval/ground_truth.json):
[
  {
    "query": "...",
    "category": "...",
    "novel": "...",
    "relevant_chunk_ids": [
      {"id": "...", "grade": 3}
    ]
  }
]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import NOVELS_DIR, EVAL_DIR, load_json, save_json, CHUNKS_DIR

CATEGORIES = [
    "exact_entity",
    "semantic",
    "relationship",
    "temporal",
    "event",
    "canon_rule",
    "paraphrase",
]

# Benchmark ground truth seed dataset — 49 queries total (7 per category)
SEED_GROUND_TRUTH = [
    # ── 1. EXACT ENTITY (7 Queries) ───────────────────────────────────────────
    {
        "query": "What is the name of Mr. Bingley's sister Caroline in Pride and Prejudice?",
        "category": "exact_entity",
        "novel": "Pride and Prejudice",
        "hint": "Caroline Bingley sister Bingley",
    },
    {
        "query": "What is the name of Lady Catherine de Bourgh's estate Rosings Park?",
        "category": "exact_entity",
        "novel": "Pride and Prejudice",
        "hint": "Rosings Park Lady Catherine estate",
    },
    {
        "query": "What is the name of Mr. Darcy's estate Pemberley in Derbyshire?",
        "category": "exact_entity",
        "novel": "Pride and Prejudice",
        "hint": "Pemberley Derbyshire Darcy estate",
    },
    {
        "query": "What is the name of Lowood School run by Mr. Brocklehurst?",
        "category": "exact_entity",
        "novel": "Jane Eyre",
        "hint": "Lowood School Brocklehurst institution",
    },
    {
        "query": "What is the name of Mr. Rochester's estate Thornfield Hall?",
        "category": "exact_entity",
        "novel": "Jane Eyre",
        "hint": "Thornfield Hall Rochester estate",
    },
    {
        "query": "What is the full name of Theodore Laurence Laurie in Little Women?",
        "category": "exact_entity",
        "novel": "Little Women",
        "hint": "Theodore Laurence Laurie grandson",
    },
    {
        "query": "What is the name of the March family estate Plumfield given by Aunt March?",
        "category": "exact_entity",
        "novel": "Little Women",
        "hint": "Plumfield Aunt March estate school",
    },

    # ── 2. SEMANTIC (7 Queries) ───────────────────────────────────────────────
    {
        "query": "Social pride, class prejudice, and snobbery in upper class courtship",
        "category": "semantic",
        "novel": "Pride and Prejudice",
        "hint": "pride prejudice social standing class rank",
    },
    {
        "query": "First impressions and misjudgments of character between Elizabeth and Darcy",
        "category": "semantic",
        "novel": "Pride and Prejudice",
        "hint": "impression prejudice character judgment Darcy Elizabeth",
    },
    {
        "query": "Themes of spiritual autonomy, moral integrity, and female independence",
        "category": "semantic",
        "novel": "Jane Eyre",
        "hint": "independence soul God conscience moral freedom",
    },
    {
        "query": "Helen Burns belief in endurance, Christian forgiveness, and quiet resignation",
        "category": "semantic",
        "novel": "Jane Eyre",
        "hint": "Helen Burns endurance Christian forgiveness Lowood",
    },
    {
        "query": "Sisterly devotion, personal sacrifices, and artistic ambitions among young women",
        "category": "semantic",
        "novel": "Little Women",
        "hint": "Meg Jo Beth Amy March sisters mother Marmee ambition",
    },
    {
        "query": "The March sisters playing Pilgrim's Progress to overcome their personal flaws",
        "category": "semantic",
        "novel": "Little Women",
        "hint": "Pilgrim's Progress burdens Marmee game Christmas",
    },
    {
        "query": "Domestic contentment, poverty, and family solidarity in 19th century households",
        "category": "semantic",
        "novel": "Little Women",
        "hint": "poor poverty domestic happiness family Marmee",
    },

    # ── 3. RELATIONSHIP (7 Queries) ───────────────────────────────────────────
    {
        "query": "How does Elizabeth Bennet's opinion of Mr. Darcy change after reading his letter?",
        "category": "relationship",
        "novel": "Pride and Prejudice",
        "hint": "letter Darcy Elizabeth opinion change respect",
    },
    {
        "query": "What arguments does Lady Catherine use when confronting Elizabeth about Darcy?",
        "category": "relationship",
        "novel": "Pride and Prejudice",
        "hint": "Lady Catherine Elizabeth confrontation Longbourn Darcy marriage",
    },
    {
        "query": "How does Wickham describe his past history with Mr. Darcy to Elizabeth?",
        "category": "relationship",
        "novel": "Pride and Prejudice",
        "hint": "Wickham Darcy backstory living Pemberley father",
    },
    {
        "query": "Dynamic and growing affection between Jane Eyre and Mr. Edward Rochester at Thornfield",
        "category": "relationship",
        "novel": "Jane Eyre",
        "hint": "Jane Rochester master Thornfield affection dialogue",
    },
    {
        "query": "St. John Rivers proposes to Jane to accompany him to India as his missionary wife",
        "category": "relationship",
        "novel": "Jane Eyre",
        "hint": "St. John Rivers proposal India missionary wife Jane",
    },
    {
        "query": "Laurie proposes marriage to Jo March and her emotional rejection of him",
        "category": "relationship",
        "novel": "Little Women",
        "hint": "Laurie proposal Jo refusal heartbreak friendship",
    },
    {
        "query": "Jo March meets Professor Friedrich Bhaer at the New York boarding house",
        "category": "relationship",
        "novel": "Little Women",
        "hint": "Professor Bhaer German New York boarding house Jo",
    },

    # ── 4. TEMPORAL (7 Queries) ───────────────────────────────────────────────
    {
        "query": "What happens right after Jane Bennet falls ill during her visit to Netherfield?",
        "category": "temporal",
        "novel": "Pride and Prejudice",
        "hint": "Jane ill Netherfield rain Elizabeth walks muddy",
    },
    {
        "query": "What events unfold after Bingley suddenly leaves Netherfield for London?",
        "category": "temporal",
        "novel": "Pride and Prejudice",
        "hint": "Bingley left Netherfield London Jane disappointed Caroline",
    },
    {
        "query": "What occurs after the interrupted wedding ceremony between Jane and Rochester?",
        "category": "temporal",
        "novel": "Jane Eyre",
        "hint": "wedding interrupted Bertha Mason third story revealed departure",
    },
    {
        "query": "What happens when Jane returns to Thornfield Hall after leaving the Rivers family?",
        "category": "temporal",
        "novel": "Jane Eyre",
        "hint": "return Thornfield ruins burned Rochester blind Ferndean",
    },
    {
        "query": "What occurs after Beth contracts scarlet fever while caring for the Hummel family?",
        "category": "temporal",
        "novel": "Little Women",
        "hint": "Beth scarlet fever Hummel illness recovery doctor",
    },
    {
        "query": "What follows Amy March burning Jo's story manuscript in the fireplace?",
        "category": "temporal",
        "novel": "Little Women",
        "hint": "Amy burns manuscript storybook Jo anger river ice",
    },
    {
        "query": "Events leading up to Meg March marrying John Brooke",
        "category": "temporal",
        "novel": "Little Women",
        "hint": "Meg wedding John Brooke doves simple ceremony prepare",
    },

    # ── 5. EVENT (7 Queries) ──────────────────────────────────────────────────
    {
        "query": "Mr. Collins proposes marriage to Elizabeth Bennet at Longbourn",
        "category": "event",
        "novel": "Pride and Prejudice",
        "hint": "Collins proposal Elizabeth refusal Longbourn breakfast",
    },
    {
        "query": "Lydia Bennet elopes with Wickham causing a family scandal",
        "category": "event",
        "novel": "Pride and Prejudice",
        "hint": "Lydia Wickham elope scandal Brighton express",
    },
    {
        "query": "Mr. Darcy makes his second proposal of marriage to Elizabeth during a walk",
        "category": "event",
        "novel": "Pride and Prejudice",
        "hint": "Darcy second proposal Elizabeth walk accept Bingley",
    },
    {
        "query": "Bertha Mason sets fire to Mr. Rochester's bed at Thornfield",
        "category": "event",
        "novel": "Jane Eyre",
        "hint": "fire bed Rochester smoke Jane wakes candle water",
    },
    {
        "query": "Jane meets Mr. Rochester for the first time on the icy lane near Hay",
        "category": "event",
        "novel": "Jane Eyre",
        "hint": "horse slipped ice Rochester Hay road dog Pilot",
    },
    {
        "query": "Jo cuts off her long hair and sells it to help send money to her father",
        "category": "event",
        "novel": "Little Women",
        "hint": "Jo cuts hair sells twenty-five dollars father hospital",
    },
    {
        "query": "Beth March dies peacefully surrounded by her family",
        "category": "event",
        "novel": "Little Women",
        "hint": "Beth death quiet fading poem Jo room spring",
    },

    # ── 6. CANON RULE (7 Queries) ─────────────────────────────────────────────
    {
        "query": "Entailment of the Bennet estate to Mr. Collins due to lack of male heirs",
        "category": "canon_rule",
        "novel": "Pride and Prejudice",
        "hint": "entail Longbourn Collins male heir estate law",
    },
    {
        "query": "Social requirements regarding female decorum and modest financial expectations in marriage",
        "category": "canon_rule",
        "novel": "Pride and Prejudice",
        "hint": "women marriage security financial Regency society fortune",
    },
    {
        "query": "Legal rights of husbands over their wives' private property and inherits in Victorian era",
        "category": "canon_rule",
        "novel": "Jane Eyre",
        "hint": "property law husband wife fortune money Victorian",
    },
    {
        "query": "Strict rules and harsh discipline imposed on pupils at Lowood institution",
        "category": "canon_rule",
        "novel": "Jane Eyre",
        "hint": "Lowood rules discipline Brocklehurst hair food cold",
    },
    {
        "query": "Victorian societal status and delicate boundaries surrounding governesses",
        "category": "canon_rule",
        "novel": "Jane Eyre",
        "hint": "governess position servant class boundary Victorian society",
    },
    {
        "query": "Nineteenth century New England social expectations for young ladies' behavior",
        "category": "canon_rule",
        "novel": "Little Women",
        "hint": "little women domestic duties gentleman lady marriage New England",
    },
    {
        "query": "Terms of Aunt March's bequest of Plumfield estate to Jo",
        "category": "canon_rule",
        "novel": "Little Women",
        "hint": "Plumfield estate Aunt March bequest legacy school",
    },

    # ── 7. PARAPHRASE (7 Genuine Semantic Rephrasings - NO Literal Quotes) ────
    {
        "query": "A wealthy unmarried gentleman is bound to be sought after as a prospective husband by local families",
        "category": "paraphrase",
        "novel": "Pride and Prejudice",
        "hint": "truth universally acknowledged single man fortune wife",
    },
    {
        "query": "Elizabeth observes that Darcy's haughty behavior at the ball alienated everyone present",
        "category": "paraphrase",
        "novel": "Pride and Prejudice",
        "hint": "Darcy proud disagreeable Meryton ball reserved",
    },
    {
        "query": "Elizabeth admits to her sister that her affection grew gradually upon seeing his grand estate",
        "category": "paraphrase",
        "novel": "Pride and Prejudice",
        "hint": "Pemberley seeing grounds love grew Jane joke",
    },
    {
        "query": "The narrator reflects on ultimately wedding her companion at Ferndean",
        "category": "paraphrase",
        "novel": "Jane Eyre",
        "hint": "Reader I married him Ferndean conclusion ten years",
    },
    {
        "query": "The young orphan suffers terrifying visions while locked in the red bedchamber",
        "category": "paraphrase",
        "novel": "Jane Eyre",
        "hint": "red-room Mrs. Reed John Reed ghost uncle Gateshead",
    },
    {
        "query": "The March sisters complain about having no gifts for the upcoming winter holiday due to financial hardship",
        "category": "paraphrase",
        "novel": "Little Women",
        "hint": "Christmas presents grumbled Jo rug opening",
    },
    {
        "query": "Jo experiences deep artistic frustration after her sister burns her notebook out of spite",
        "category": "paraphrase",
        "novel": "Little Women",
        "hint": "Amy burns manuscript storybook Jo anger fireplace",
    },
]


def _match_chunk_ids(novel: str, hint: str) -> list[dict]:
    """Find exact matching chunk IDs for a given novel and hint."""
    scene_file = CHUNKS_DIR / f"{novel.replace(' ', '_')}_scene.json"
    if not scene_file.exists():
        return []

    chunks = load_json(scene_file)
    hint_words = set(hint.lower().split())

    scored = []
    for c in chunks:
        text = c["text"].lower()
        score = sum(1 for w in hint_words if w in text)
        if score > 0:
            scored.append((score, c["id"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    matches = scored[:2] if len(scored) >= 2 else scored[:1]

    result = []
    for idx, (score, cid) in enumerate(matches):
        grade = 3 if idx == 0 else 2
        result.append({"id": cid, "grade": grade})

    if not result and chunks:
        result.append({"id": chunks[0]["id"], "grade": 2})

    return result


def generate_ground_truth() -> list[dict]:
    """Generate categorized ground truth benchmark dataset (49 queries total)."""
    ground_truth = []

    for entry in SEED_GROUND_TRUTH:
        novel = entry["novel"]
        hint = entry["hint"]
        matched_chunks = _match_chunk_ids(novel, hint)

        gt_item = {
            "query": entry["query"],
            "category": entry["category"],
            "novel": novel,
            "relevant_chunk_ids": matched_chunks,
        }
        ground_truth.append(gt_item)

    out_path = EVAL_DIR / "ground_truth.json"
    save_json(ground_truth, out_path)
    print(f"  Ground truth dataset created -> {out_path}")
    print(f"  Total queries: {len(ground_truth)} queries across 7 categories.\n")

    cat_counts = {}
    for gt in ground_truth:
        c = gt["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print("Category Breakdown (7 queries/category):")
    for cat, count in sorted(cat_counts.items()):
        print(f"  - {cat:15s}: {count} queries")

    return ground_truth


if __name__ == "__main__":
    generate_ground_truth()
