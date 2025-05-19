import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retriever import Retriever

# Set file path and query list here
file_path = "homework-week-3/docs/document1.txt"
queries = [
    "What is the main idea of the text?",
    "Who is the main character?"
]

def run(file_path, queries, chunk_size=100, chunk_overlap=20, top_k=3):
    retriever = Retriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever.add_documents([file_path])
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.query(query, top_k=top_k)
        if not results:
            print("No results found.")
        for i, chunk in enumerate(results):
            print(f"Result {i + 1}:\n{chunk}\n")
        print("-" * 50)

if __name__ == "__main__":
    run(file_path, queries)

