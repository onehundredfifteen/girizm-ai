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

# save embeddings in db
def save_embedding(cur, embedding_id, embedding):
    
    print(f"Adding embedding {embedding_id}({len(embedding)}): {embedding[:90]}...")
    cur.execute("""
    INSERT INTO verses(id, verse, embedding)
    VALUES (?, ?, ?)
    """, (
        embedding_id,
        embedding,
        json.dumps(embedding)
    ))

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
    embeddings.append(calculate_embedding(line))
for chunk in chunks:
    embeddings.append(calculate_embedding(chunk))

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

for i, c in enumerate(embeddings):
    save_embedding(cur, i, c)

print(f">> Added {len(embeddings)} embeddings")

db.commit()
db.close()