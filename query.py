import sys

import ollama
import sqlite3
import json
import math


MODEL = "nomic-embed-text"
CHAT_MODEL = "girizm-ai-strict"
DBFILE = "embeddings-exp.db"

if(len(sys.argv) < 2):
    print("Usage: python query.py <query>|1|2|3")
    sys.exit(1)

QUERY = "Kto to Gira i czym jest Girizm?"

query_arg = str(sys.argv[1])
if(query_arg == "1"):
    QUERY = "Co radzi Girizm, gdy ktoś jest smutny?"
elif(query_arg == "2"):
    QUERY = "Co radzi Girizm, gdy ktoś jest zlosliwy?"
elif(query_arg == "3"):
    QUERY = "Dwa złote czy to dużo dla wyznawcy Giry?"
else:
    QUERY = query_arg    

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

print(f"\"{QUERY}\"")
print(f"{CHAT_MODEL}: Looking in memory...")
context = get_context()

print(f"{CHAT_MODEL}: Connecting the dots...")
response = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "user", "content": "Context: " + context},
        {"role": "user", "content": QUERY}
    ]
)

answer = response["message"]["content"]

print(f"{CHAT_MODEL} says: {answer}")

db.close()