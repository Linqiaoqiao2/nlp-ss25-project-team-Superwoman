# NLP-Homework-1-semantic_search_faiss

## Analysis of query results
- We used 4 different formulations of the same question: "Why do people use TikTok?". The model gave the same top 3 results for each of the queries, which means the model seems to conform to the queries correctly.

- The only difference is for query 2, where the order of position 2 and 3 in the results has switched. The potential explanation is that the query asks about **watching videos** and the result that moved up from position 3 to 2 is about **sharing videos**, which means there is a bigger overlap of the words used compared to the result in position 3.

## How Vector Search Works

- Vector search is a method of finding semantically similar content based on dense vector representations. First, natural language texts are transformed into high-dimensional vectors using pre-trained models like `all-MiniLM-L6-v2`. These vectors capture the meaning of the text.

- FAISS (Facebook AI Similarity Search) is then used to index and efficiently search these vectors by computing distances between them (e.g., using L2 or cosine distance). When a user query is converted into a vector, FAISS retrieves the most similar entries from the index, enabling semantic search beyond keyword matching.
