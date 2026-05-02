import csv
import re
from pathlib import Path

input_path = Path("data/raw_shakespeare/Sonnet.txt")
output_path = Path("data/shakespeare_full.csv")

text = input_path.read_text(encoding="utf-8-sig")

# Remove obvious header lines
text = text.replace("THE SONNETS", "", 1)
text = text.replace("by William Shakespeare", "", 1).strip()

lines = text.splitlines()

poems = []
current = []

def flush_current():
    global current
    if current:
        poem_lines = [ln.rstrip() for ln in current if ln.strip()]
        if poem_lines:
            poems.append("\n".join(poem_lines))
        current = []

for line in lines:
    stripped = line.strip()

    # New sonnet starts when a non-indented line begins with a capital letter
    # after we already have 14-ish lines collected and the previous line was blank-ish enough.
    if stripped == "":
        if current:
            current.append("")
        continue

    # If we already have a poem collected and encounter a likely new opening line,
    # start a new poem.
    if current:
        nonempty = [ln for ln in current if ln.strip()]
        if (
            len(nonempty) >= 12
            and not line.startswith(" ")
            and stripped
            and stripped[0].isupper()
        ):
            # Stronger signal: opening lines usually don't end with punctuation like prior line continuation
            prev = nonempty[-1].strip()
            if prev.endswith((".", "!", "?", ":", ";", ",")):
                flush_current()

    current.append(line)

flush_current()

rows = []
for i, poem in enumerate(poems, start=1):
    poem_lines = [ln.strip() for ln in poem.splitlines() if ln.strip()]
    if not poem_lines:
        continue
    title = poem_lines[0]
    poem_text = "\n".join(poem_lines)
    rows.append({
        "sonnet_number": i,
        "title": title,
        "text": poem_text
    })

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["sonnet_number", "title", "text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} sonnets to {output_path}")
if rows:
    print("First title:", rows[0]["title"])
    print("Last title:", rows[-1]["title"])
