import ollama
import sqlite3
import json
import math
import string

MODEL = "nomic-embed-text"
GIRYZM_TROUTH_SOURCE = "giryzm-tokenized.txt"


lines = []
verses = []

with open(GIRYZM_TROUTH_SOURCE, "r") as file:
    contents = file.read()
    lines = contents.split("\n")
    for line in lines:
        s = line.translate(str.maketrans({'.': '$', ';': '$','\n': '$'}))
        verses.extend(s.split())

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
for line in verses:
    last_verse = "1"
    subverse_counter = 1

    parts = line.split('|')
    if len(parts) == 2:
        verse_id, verse_text = parts
    else:
        verse_text = parts[0]
        verse_id = last_verse + f".{subverse_counter}"
        subverse_counter += 1

    print()(f"Przetwarzanie wersetu {verse_id}")    

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