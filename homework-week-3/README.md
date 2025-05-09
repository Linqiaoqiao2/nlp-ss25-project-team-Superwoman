# Semantic Document Retriever

This project builds on the semantic retrieval demo we created last week, where we experimented with sentence embeddings and FAISS using a small set of example sentences.


## What's Improved Since Last Week

This version turns that early experiment into a more complete and reusable retrieval system. Compared to last week, we've made several key improvements:

- We now support **real documents**, including `.txt`, `.md`, and `.pdf` files
- Instead of working with fixed sentences, we **automatically split documents into chunks** with overlap to preserve context
- All core functions are wrapped into a clean and reusable **`Retriever` class**
- The system includes a simple `.query()` interface for semantic search
- We added support for **saving and loading** the FAISS index
- A basic **test script** and a structured project layout were added
- This `README.md` explains how everything works and how to use it

While last week's work helped us understand the basics of embeddings and vector search, this week's version is more practical and modular-something we can reuse or build on in future projects.


## How Vector Search Works

1. Input text is split into chunks with overlap to preserve context

2. Each chunk is converted into a dense vector (embedding)

3. These vectors are indexed using FAISS

4. Queries are also embedded and compared to the index using vector distance (L2)

5. Top matching chunks are returned to the user

##  Usage Instructions

To use this project, follow the steps below.

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu pymupdf transformers

# 2. Load documents and build index
# (Python code)
python
from retriever import Retriever

r = Retriever()
r.add_documents(["docs/example.txt", "docs/ai_human.md"])

# 3. Run a query
results = r.query("Can AI think like humans?", top_k=3)
for res in results:
    print(res)

# 4. Save or load the FAISS index
r.save("my_index")   # Save to disk
r.load("my_index")   # Load from disk

##  Reflections and Thoughts

We tested our system using both English and German documents. Currently, it only supports querying English documents in English and German documents in German. In the future, we could build on this foundation to enable cross-lingual retrieval.
