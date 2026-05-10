import ollama
import sqlite3
import json
import math
import string

MODEL = "nomic-embed-text"
CHAT_MODEL = "girizm-ai"

# --- baza ---
db = sqlite3.connect("embeddings.db")
cur = db.cursor()

# --- cosine similarity ---
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot / (norm_a * norm_b)

# --- test ---
query = "Kim jest Gira?"

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

context = ""

for score, verse_id, verse_text in results[:5]:
    context += f"{verse_text}."

response = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        #{"role": "system", "content": "You are a Senior Priest of the god called Gira. You know the verses of the holy book called Girizm by heart. You are a wise and helpful guide to those who seek the truth."},
        {"role": "user", "content": f"Na podstawie tego kontekstu: {context} Odpowiedz na pytanie: {query}"}
    ]
)

print(response)
answer = response["message"]["content"]

print("Odpowiedź:", answer)

db.close()