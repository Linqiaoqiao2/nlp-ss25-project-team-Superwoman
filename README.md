# NLP-Homework-1

 ## Summary

 ### ? How Vector Search Works

Vector search is a method of finding semantically similar content based on dense vector representations. First, natural language texts are transformed into high-dimensional vectors using pre-trained models like `all-MiniLM-L6-v2`. These vectors capture the meaning of the text.

FAISS (Facebook AI Similarity Search) is then used to index and efficiently search these vectors by computing distances between them (e.g., using L2 or cosine distance). When a user query is converted into a vector, FAISS retrieves the most similar entries from the index, enabling semantic search beyond keyword matching.
