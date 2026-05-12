import ollama
import sqlite3
import json

DBFILE = "embeddings.db"
MODEL = "nomic-embed-text"
GIRYZM_TRUTH_SOURCE = "giryzm-tokenized.txt"

# --- some methods for processing data etc
# save embeddings in db
def add_embedding(cur, embedding_id, memory_text):
    response = ollama.embeddings(
        model=MODEL,
        prompt=memory_text.strip()
    )

    embedding = response["embedding"]
    print(f"Adding embedding {embedding_id}({len(memory_text)}): {memory_text[:90]}...")
    cur.execute("""
    INSERT INTO verses(id, verse, embedding)
    VALUES (?, ?, ?)
    """, (
        embedding_id,
        memory_text,
        json.dumps(embedding)
    ))

# bulk load
def bulk_add_embeddings(cur, start_idx, content):
    for i, c in enumerate(content):
        add_embedding(cur, start_idx + i, c)
    
    return start_idx + len(content)

# generate chunks of sentences with some overlap
def chunk_sentences(sentences, max_sentences, overlap):
    chunks = []
    start = 0

    while start < len(sentences):
        end = start + max_sentences

        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

        start = end - overlap

    return chunks

# --- start of the main code

# data
lines = [] # raw lines from file
verses = [] # sentences split from lines
chunks = [] # chunks of sentences

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


start_idx = bulk_add_embeddings(cur, 0, lines)
print(f">> Added embeddings for {len(lines)} lines.")
#start_idx = bulk_add_embeddings(cur, start_idx, verses)
#print(f">> Added embeddings for {len(verses)} verses.")
start_idx = bulk_add_embeddings(cur, start_idx, chunks)
print(f">> Added embeddings for {len(chunks)} overlapping chunks.")

db.commit()
db.close()