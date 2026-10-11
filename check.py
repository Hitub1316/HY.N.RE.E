import json, random, hashlib
from pathlib import Path

base = Path("data/chunks")
files = {
    "Pride and Prejudice": base / "Pride_and_Prejudice_scene.json",
    "Jane Eyre": base / "Jane_Eyre_scene.json",
    "Little Women": base / "Little_Women_scene.json",
}
pre = json.load(open("cache/preambles.json", encoding="utf-8"))

h2c = {}
for novel, p in files.items():
    chunks = json.load(open(p, encoding="utf-8"))
    print(novel, "chunks:", len(chunks))
    for c in chunks:
        h = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        h2c[h] = (novel, c)

texts = [v["preamble"] if isinstance(v, dict) else v for v in pre.values()]
print("Template preambles (start with In '):",
      sum(t.startswith("In '") for t in texts))
print("Preambles matched to a chunk:",
      sum(h in h2c for h in pre), "of", len(pre))

for h in random.sample([h for h in pre if h in h2c], 3):
    novel, c = h2c[h]
    v = pre[h]
    print("\n", novel, c["id"], "|", c.get("chapter"))
    print("CHUNK:", c["text"][:300].replace("\n", " "))
    print("PREAMBLE:", v["preamble"] if isinstance(v, dict) else v)

print("Preambles mentioning 'chapter':",
      sum("chapter" in t.lower() for t in texts))