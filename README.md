```markdown
# RAG Project �C Summer Semester 2025

> **Retrieval-Augmented Generation** baseline implementation and roadmap for ongoing development.

---

## ? Overview

This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Currently under active development.

---

## Retriever Module

The `Retriever` class in `retriever.py` provides end-to-end semantic search over text and PDF documents:

1. **Load documents** (`.txt`, `.md`, `.pdf`)  
2. **Chunk text** into overlapping token windows  
3. **Embed chunks** with a SentenceTransformer  
4. **Index embeddings** in FAISS for fast similarity search  
5. **Query** the index to return top-K relevant chunks  
6. **Save** / **Load** the index and chunk metadata to/from disk  

### Key Features

- **Multi-format support**: `.txt`, `.md`, `.pdf`  
- **Configurable chunking**: adjustable `chunk_size` and `chunk_overlap` parameters  
- **Custom embeddings**: plug in any SentenceTransformer model  
- **Fast similarity search**: powered by FAISS `IndexFlatL2`  
### Implementation Details

#### Document Loading

- **`.txt` / `.md`**: UTF-8 text reading  
- **`.pdf`**: page-by-page text extraction via [PyMuPDF](https://pymupdf.readthedocs.io/)  

#### Chunking Strategy

- **Tokenization**: using `AutoTokenizer` (WordPiece)  
- **Sliding window**: generate overlapping chunks to preserve context  

#### Embeddings

- **Batch encoding**: use `SentenceTransformer.encode` to convert chunks into vector embeddings  

#### Indexing

- **FAISS**: use `IndexFlatL2` for L2-distance based nearest neighbor search  

###  Usage Instructions

To use this project, follow the steps below.

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu pymupdf transformers

# 2. Load documents and build index
# (Python code)
python
from retriever import Retriever

r = Retriever(embedding_model="all-MiniLM-L6-v2", chunk_size=500, overlap=100)
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







---


## ? Team Members

| Role        | Name             | GitHub Username      |
|-------------|------------------|----------------------|
| Member      | Mengmeng Yu      | [Linqiaoqiao2]       |
| Member      | Wenhui Deng      | [deng-wenhui]        |
| Member      | Subhasri Suresh  | [subhasri-suresh]    |
```

---


