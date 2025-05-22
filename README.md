# RAG Project - Summer Semester 2025

> **Retrieval-Augmented Generation** baseline implementation and roadmap for ongoing development.

---

## Overview

This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Currently under active development.
---

## Folder Structure
<pre> 
project-root/
.
├── data
│   ├── Germany's foreign policy?.txt
│   ├── Rotkaeppchen.pdf
│   ├── ai_human.md
│   └── darwins_theory.txt
├── generator
│   └── generator.py
├── llms
│   ├── README.MD
│   └── ** your model goes here **
├── logs
│   └── retriever.log
├── pipeline.py
├── retriever
│   └── retriever.py
├── test
│   ├── input
|   |   └── questions.txt
│   ├── outputs
|   |   └── answers.txt
│   ├── RAGPipelineTest.py
│   └── test_retriever.py
└── util
    ├── fileUtil.py
    └── logger_config.py


</pre>
 1. **To run the RAGPipeline, run** python test/RAGPipelineTest.py
 2. **To test with new question, add your question to:** test/input/questions.txt
 3. **Output file:** test/output/answers.txt

 
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

* **Multi-format support**: `.txt`, `.md`, `.pdf`
* **Configurable chunking**: adjustable `chunk_size` and `chunk_overlap` parameters
* **Custom embeddings**: plug in any SentenceTransformer model
* **Fast similarity search**: powered by FAISS `IndexFlatL2`

### Implementation Details

#### Document Loading

* **`.txt` / `.md`**: UTF-8 text reading
* **`.pdf`**: page-by-page text extraction via [PyMuPDF](https://pymupdf.readthedocs.io/)

#### Chunking Strategy

* **Tokenization**: using `AutoTokenizer` (WordPiece)
* **Sliding window**: generate overlapping chunks to preserve context

#### Embeddings

* **Batch encoding**: use `SentenceTransformer.encode` to convert chunks into vector embeddings

#### Indexing

* **FAISS**: use `IndexFlatL2` for L2-distance based nearest neighbor search

---

## Usage Instructions

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu pymupdf transformers

# 2. Load documents and build index
python - <<EOF
from retriever import Retriever

r = Retriever(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=500,
    overlap=100
)
r.add_documents([
    "docs/example.txt",
    "docs/ai_human.md"
])
EOF

# 3. Run a query
python - <<EOF
results = r.query("Can AI think like humans?", top_k=3)
for res in results:
    print(res)
EOF

# 4. Save or load the FAISS index
python - <<EOF
r.save("my_index")   # Save to disk
r.load("my_index")   # Load from disk
EOF
```

---

## Generator Module

The `Generator` class in `generator.py` utilizes a local LlamaCpp model for generating answers based on retrieved context, which is useful for question-answering systems.

1. **Load llm model**  from a given file path
2. **Generate answers** by combining a user query with relevant context chunks using a customizable prompt template
3. **Handle errors** to manage issues during model loading

## Dependencies

1. **LlamaCpp**: The library used for loading and interacting with the LlamaCpp model.
2. **os**: Standard library for file path operations.
3. **sys**: The model used to manipulate the Python runtime environment, access command-line arguments, and handle standard input/output streams.
4. **List**: In the generate_answer method, List[str] specifies that the context_chunks parameter should be a list of strings.
5. **StreamingStdOutCallbackHandler**: It handles streaming output from the LlamaCpp model, allowing the generated text to be printed to standard output in real-time as it is produced.

## Usage Instructions

```bash

# 1. Provide the file path to the LlamaCpp model
python - <<EOF
            self.llm = LlamaCpp(
                model_path = filePath,
                n_gpu_layers=0,
                n_batch=64,
                n_ctx=1024,
                f16_kv=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False,
            )
EOF

# 2. Generate answers with prompts
python - <<EOF
def generate_answer(self, query: str, context_chunks: List[str], prompt_template: str) -> str:
        context = "\n".join(context_chunks)
        prompt = prompt_template.format(context=context, query=query)
        return self.llm.invoke(prompt)
EOF

# 3. Handle errors
python - <<EOF
def __init__(self, filePath):
        try:
            print(f"? Loading model from: {filePath}")
            current_dir = os.path.dirname(__file__)
        except Exception as e:
            print(f"? Failed to load model: {e}")
            raise
EOF
```

---

## Reflections and Thoughts

We tested our system using both English and German documents. Currently, it only supports querying English documents in English and German documents in German. In the future, we could build on this foundation to enable cross-lingual retrieval.

We can validate input parameters for the Generator class to make sure the output from the model conforms to the expected results. For example, the text chunks should not be an empty list or the generated answers should be in the format of string.


---


## Team Members

| Role   | Name            | GitHub Username |
| ------ | --------------- | --------------- |
| Member | Mengmeng Yu     | Linqiaoqiao2    |
| Member | Wenhui Deng     | deng-wenhui     |
| Member | Subhasri Suresh | subhasri-suresh |

---

