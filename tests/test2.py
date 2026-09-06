from pathlib import Path

novel_path = Path("novels") / "Pride and Prejudice.txt"
if novel_path.exists():
    text = novel_path.read_text(encoding="utf-8")
    print("First 200 chars of cleaned P&P text:")
    print(repr(text[:200]))
else:
    print(f"File {novel_path} not found.")