import ollama
import sqlite3
import json
import math
import string

MODEL = "nomic-embed-text"
GIRYZM_TROUTH_SOURCE = "giryzm-tokenized.txt"


lines = []
verses = []
verse_keys = []

with open(GIRYZM_TROUTH_SOURCE, "r") as file:
    contents = file.read()
    lines = contents.split("\n")[1:]  # skip header

    for line in lines:
        s = line.translate(str.maketrans({'.': '$', ';': '$','\n': '$'}))
        v = s.split('$')
        verses.extend(v)
        verse_keys.append(len(v))

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

cur.execute("""
DELETE FROM verses ;
""")

# --- podział na wersety i embedding ---
verse_id = 0
subverse_counter = 0
last_verse_key = "1"
line_key = 0
for v in verses:
    parts = v.split('|')
    
    if len(parts) == 2:
        verse_key, verse_text = parts
        last_verse_key = verse_key
        subverse_counter = verse_keys[line_key] - 1
        line_key += 1
    else:
        subverse_counter -= 1
        subkey = verse_keys[line_key] - subverse_counter
        verse_key = f"{last_verse_key}.{subkey}"
        verse_text = parts[0]
        

    print(f"Przetwarzanie wersetu {verse_key}:{verse_text}")    

    pro = f"{verse_key.strip()} {verse_text.strip()}"

    if(pro.strip() == ""):
        print(f"PSkip wersetu {verse_key}:{verse_text}") 
        continue

    response = ollama.embeddings(
        model=MODEL,
        prompt=pro.strip()
    )

    embedding = response["embedding"]

    cur.execute("""
    INSERT INTO verses(id, verse, embedding)
    VALUES (?, ?, ?)
    """, (
        verse_id,
        verse_text,
        json.dumps(embedding)
    ))

    verse_id+=1

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