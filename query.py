import ollama
import sqlite3
import json
import math
import string

MODEL = "nomic-embed-text"
CHAT_MODEL = "girizm-ai"
DBFILE = "embeddings.db"

QUERY = "Kim jest Gira?"

# --- setup ---
db = sqlite3.connect(DBFILE)
cur = db.cursor()

# --- cosine similarity ---
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot / (norm_a * norm_b)

def get_context():
    query_embedding = ollama.embeddings(
        model=MODEL,
        prompt=QUERY
    )["embedding"]

    results = []

    for row in cur.execute("SELECT id, verse, embedding FROM verses"):
        verse_id, verse_text, emb_json = row

        embedding = json.loads(emb_json)

        score = cosine_similarity(query_embedding, embedding)

        results.append((score, verse_id, verse_text))

    results.sort(reverse=True)

    context = ""

    for score, verse_id, verse_text in results[:5]:
        log_verse = verse_text[:90]
        if len(verse_text) > 90:
            log_verse += "..."

        print(f"Found chunk #{verse_id} with score {score:.4f}: {log_verse}")
        context += f"{verse_text}.\n"

    return context

print(f"{QUERY}")
print("Looking in memory...")
context = get_context()

print("Connecting the dots...")
response = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "user", "content": "Context: " + context},
        {"role": "user", "content": f"{QUERY}"}
    ]
)

answer = response["message"]["content"]

print("Odpowiedź:", answer)

db.close()