# NLP-Homework-1-semantic_search_faiss

## Analysis of Query Results
- We used 4 different formulations of the same question: "Why do people use TikTok?". The model gave the same top 3 results for each of the queries, which means the model seems to conform to the queries correctly.

- The only difference is for query 2, where the order of position 2 and 3 in the results has switched. The potential explanation is that the query asks about **watching videos** and the result that moved up from position 3 to 2 is about **sharing videos**, which means there is a bigger overlap of the words used compared to the result in position 3.

## How Vector Search Works

- Vector search is a method of finding semantically similar content based on dense vector representations. First, natural language texts are transformed into high-dimensional vectors using an embedding model. In our case we used the pre-trained model `all-MiniLM-L6-v2` from the sentence-transformers library. These vectors capture the meaning of the text, which can be compared using cosine similarity. If two sentences have a similar meaning, their vectors will be close together in the embedding space. You can see this in a PCA visualization.

- The next step is similarity search. FAISS (Facebook AI Similarity Search) helps by building a fast index that makes it much quicker to search for the most similar vectors. We build an index for sentence vectors using FAISS. When a query is provided, its vector is compared against the stored vectors to find the closest matches.
	- For example:
	When a user asks a question ("Why do people use TikTok?"), the system:
		1. Encodes the question into a vector.
		2. Searches for the top k closest sentence vectors in the FAISS index.
        3. Returns the most relevant sentences based on vector proximity (low L2 distance).