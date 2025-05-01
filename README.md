# NLP-Homework-1-semantic_search_faiss

	
		1. The model 'all-MiniLM-L6-v2' is used from the sentence-transformers library to convert the sentences into vector representations. These vectors are then compared using cosine similarity, because if two sentences mean the same thing, their vectors will be close together in the embedding space. You can see this in a PCA visualization.
		2. Then it's about similarity search.
FAISS helps by building a fast index that makes it much quicker to search for the most similar vectors. So, we build an index for sentence vectors using FAISS. When a query is provided, its vector is compared against the stored vectors to find the closest matches.
	For example:
	When a user asks a question ("Why do people use TikTok?"), the system:
		• Encodes the question into a vector.
		• Searches for the top k closest sentence vectors in the FAISS index.
Returns the most relevant sentences based on vector proximity (low L2 distance).
