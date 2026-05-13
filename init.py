import math

import ollama
import sqlite3
import json

DBFILE = "embeddings.db"
MODEL = "nomic-embed-text"
GIRYZM_TRUTH_SOURCE = "giryzm-tokenized.txt"

# --- some methods for processing data etc

# calculate embedding
def calculate_embedding(text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=text.strip()
    )

    return response["embedding"]

# --- cosine similarity ---
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    return dot / (norm_a * norm_b)

def deduplicate(embeddings, threshold=0.95):
    
    unique_embeddings = []
    for candidate in embeddings:
        duplicate = False
        for existing in unique_embeddings:
            similarity = cosine_similarity(candidate[1], existing[1])
            if similarity > threshold:
                duplicate = True
                break

        if not duplicate:
            unique_embeddings.append(candidate)

    return unique_embeddings

# generate chunks of sentences with some overlap
def chunk_sentences(sentences, max_sentences, overlap):
    chunks = []
    start = 0

    while start < len(sentences):
        end = start + max_sentences

        chunk = ". ".join(sentences[start:end],)
        chunks.append(chunk + ".")

        start = end - overlap

    return chunks

# --- start of the main code

# data
lines = [] # raw lines from file
verses = [] # sentences split from lines
chunks = [] # chunks of sentences
embeddings = [] # list of all embeddings

# load lines from file and split into sentences
with open(GIRYZM_TRUTH_SOURCE, "r") as file:
    contents = file.read()
    lines = contents.split("\n")[1:]  # skip header
    # split sentences -> by . ; ? ! \n
    for line in lines:
        s = line.strip().translate(str.maketrans({'.':'$',';':'$','\n':'$','!':'$','?':'$'}))
        verses.extend(filter(str.strip, s.split('$')))

# generate chunks
chunks = chunk_sentences(verses, max_sentences=5, overlap=1)
chunks.extend(chunk_sentences(verses, max_sentences=8, overlap=3))

# calculate embeddings for all chunks
for line in lines:
    embeddings.append((line, calculate_embedding(line)))
for chunk in chunks:
    embeddings.append((chunk, calculate_embedding(chunk)))

print(f"{len(embeddings)} embeddings calculated")
embeddings = deduplicate(embeddings)
print(f"{len(embeddings)} semantically unique embeddings after deduplication")
for (text, embedding) in embeddings:
    print(f"initial Embedding: {text[:90]}...")
    

# database setup 
db = sqlite3.connect(DBFILE)
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS verses (
    id INTEGER PRIMARY KEY,
    verse TEXT,
    embedding TEXT
)
""")

cur.execute("""DELETE FROM verses;""")

for i, (text, embedding) in enumerate(embeddings):
    print(f"Adding embedding {i}({len(text)}): {text[:90]}...")
    cur.execute("""
    INSERT INTO verses(id, verse, embedding)
    VALUES (?, ?, ?)
    """, (
        i,
        text,
        json.dumps(embedding)
    ))

print(f">> Added {len(embeddings)} embeddings")

db.commit()
db.close()