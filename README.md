# RAG Project - Summer Semester 2025

> **🎓 University Course Chatbot – RAG-Based QA System** 

---

## Overview

This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. A chatbot designed to answer university course-related queries using website information and open-source language models.

---

## Folder Structure
<pre> 

baseline/
├── data
│   ├── cleaned_json
│   │   └── *** All cleaned data in json format from university website**
│   │   
│   └── pdfs
│       └── *** Website data in .pdf format is placed here ***
├── evaluation
│   └── evaluation_unsupervised.py
├── generator
│   └── generator.py
├── llms
│   ├── Place the suitable LLM model here (Dowloadable from https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json)
│   └── README.MD
├── logs
│   └── RAGPipeline.log
├── pipeline.py
├── requirements.txt
├── retriever
│   └── retriever_bge_m3.py
├── runPipeline.py
├── test
│   ├── F1_score_calculation.py
│   ├── input
│   │   ├── f1_qa_input.json
|   |   ├── bert_qa_input.json
│   │   └── questions.txt
│   ├── outputs
│   │   ├── answers.txt
│   │   └── f1_output.txt
│   ├── RAGPipelineTest_bge_m3.py
│   └── test_retriever.py
└── util
    ├── fileUtil.py
    └── logger_config.py



</pre>
## Steps to setup and run from the `baseline` folder
 1. **Make sure your are currently inside the /baseline folder**
 2. **Place the LLM file in the baseline/llms folder:**  https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json
 3. **To install the required packages:** Run the below commands in the terminal to install the requiered package
                ```bash
                    conda env create -f environment.yml;
                    conda activate rag-bot-test
                 ```
 
 4. **To run the chatbot:** python runPipeline.py
 5. **To access the chatbot UI:** Access the url that is displayed once the server starts. Eg: http://127.0.0.1:xxxx


## Steps to setup and run testcases
 1. **Make sure your are currently inside the /baseline folder**
 2. **To test with new question, add your question to:** test/input/questions.txt
 3. **To run the RAGPipeline with test data/questions:** python test/RAGPipelineTest_bge_m3.py
 5. **Output file:** test/output/answers.txt
 
 
---
## Usage Instructions

## Retriever Module

The `Retriever` class in `retriever.py` implements hybrid document retrieval for RAG systems, combining dense and sparse methods to improve coverage.

1. **Load and index documents** (`.txt`, `.pdf`, `.json`) with overlapping chunking  
2. **Support hybrid retrieval** using dense vectors (BGE-M3) and BM25 with weighted score fusion  
3. **Optional reranking** with a cross-encoder for more accurate results  
4. **Track sources** so each chunk can be traced back to its original file  
5. **Save and load** FAISS and BM25 indexes for reuse


### 🧩 Dependencies

The `Retriever` module depends on the following Python packages:

- sentence-transformers
- transformers
- faiss-cpu
- rank-bm25
- PyMuPDF
- loguru

To install:

```bash
pip install sentence-transformers transformers faiss-cpu rank-bm25 PyMuPDF loguru
```


### ✅ Features

- Dense + sparse retrieval with score fusion  
- Optional cross-encoder reranker  
- Overlapping chunking based on tokenizer  
- Supports `.txt`, `.md`, `.pdf`, `.json` files  
- Source tracking for each chunk  
- Easy saving and loading of FAISS/BM25 index


---

## Usage Instructions


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



## Reflections and Thoughts

We tested our system using both English and German documents. Currently, it only supports querying English documents in English and German documents in German. In the future, we could build on this foundation to enable cross-lingual retrieval.

We can validate input parameters for the Generator class to make sure the output from the model conforms to the expected results. In terms of syntax, the text chunks should not be an empty list or the generated answers should be in the format of string. In terms of semantics, the standard of consine similarity score of query and retrieved chunks can be tested differently to find out the best score to return the most suitable chunks for the preparation of answer generation.

How to handle dynamic data and realize the on-time update in our raw data automatically will be of great significance for the chatbot’s performance. With the help of it, our chatbot can always be supported by the lastest information from the webpages.


---


## Team Members

| Role   | Name            | GitHub Username |
| ------ | --------------- | --------------- |
| Member | Mengmeng Yu     | Linqiaoqiao2    |
| Member | Wenhui Deng     | deng-wenhui     |
| Member | Subhasri Suresh | subhasri-suresh |

---

