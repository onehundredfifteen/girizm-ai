import math

import ollama
import sqlite3
import json

DBFILE = "embeddings-exp.db"
MODEL = "llama3.2"

PROMPT = """Extract facts from the text.

Rules:
- Only use information from the context.
- Do not guess.
- Keep facts short.
- Remove duplicates.
- Preserve numbers, names, versions, and units.
- Ignore small talk.

Format:
- fact
- fact
- fact

Text:
{{CONTEXT}}"""

# --- some methods for processing data etc
def find_facts(verses):
    fact_idx=0
    for verse_id, verse_text in verses:
        print(f"Finding facts for verse {verse_id}({len(verse_text)}): {verse_text[:90]}...")
        response = ollama.chat(
            model=MODEL,
            prompt=PROMPT.replace("{{CONTEXT}}", verse_text.strip())
        )

        facts = response["response"].strip().split("\n")
        facts = [fact.strip("- ").strip() for fact in facts if fact.strip()]
        
        for fact in facts:
            cur.execute("""
            INSERT INTO facts(id, embedding_id, fact)
            VALUES (?, ?, ?)
            """, (
                fact_idx,
                verse_id,
                fact
            ))
            fact_idx += 1

# database setup 
db = sqlite3.connect(DBFILE)
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    embedding_id INTEGER,
    fact TEXT
)
""")

cur.execute("""DELETE FROM facts;""")

verses = []
cur.execute("SELECT id, verse, embedding FROM verses")
for row in cur.fetchall():
    verse_id, verse_text = row
    verses.append((verse_id, verse_text))
    

find_facts(verses)

db.commit()
db.close()