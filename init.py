import ollama
import sqlite3
import json
import math

MODEL = "nomic-embed-text"

TEXT = """
1|Na początku był kod.
2|Kod był z programistą.
3|I kod był dobry.
"""

# --- baza ---
db = sqlite3.connect("embeddings.db")
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS verses (
    id INTEGER PRIMARY KEY,
    verse TEXT,
    embedding TEXT
)
""")

# --- podział na wersety i embedding ---
for line in TEXT.strip().splitlines():
    verse_id, verse_text = line.split("|", 1)

    response = ollama.embeddings(
        model=MODEL,
        prompt=verse_text
    )

    embedding = response["embedding"]

    cur.execute("""
    INSERT INTO verses(id, verse, embedding)
    VALUES (?, ?, ?)
    """, (
        int(verse_id),
        verse_text,
        json.dumps(embedding)
    ))

db.commit()

# --- cosine similarity ---
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot / (norm_a * norm_b)

# --- wyszukiwanie ---
query = "programowanie"

query_embedding = ollama.embeddings(
    model=MODEL,
    prompt=query
)["embedding"]

results = []

for row in cur.execute("SELECT id, verse, embedding FROM verses"):
    verse_id, verse_text, emb_json = row

    embedding = json.loads(emb_json)

    score = cosine_similarity(query_embedding, embedding)

    results.append((score, verse_id, verse_text))

results.sort(reverse=True)

print("\nNajbardziej podobne:\n")

for score, verse_id, verse_text in results[:3]:
    print(f"{score:.4f} | {verse_id} | {verse_text}")

db.close()