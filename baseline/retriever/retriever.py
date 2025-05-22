import os
import pickle
from util.logger_config import setup_logger

from pathlib import Path
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from transformers import AutoTokenizer

logger = setup_logger("Retriever", log_file="logs/RAGPipeline.log")

class Retriever:
    """
    Retriever class for document indexing and semantic search using FAISS and SentenceTransformers.
    """
    # Initialize Retriever with embedding model and chunk parameters, so they can be reused across methods.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 200, chunk_overlap: int = 50):
        
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[str] = []
        self.index = None
    
    """
    Load document from file path and return text content(String).
    Supports .txt, .md, and .pdf formats.
    For .pdf, uses PyMuPDF to extract text.
    For .txt and .md, reads the file directly.
    Raises ValueError for unsupported file formats.
    """
    def load_document(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix in (".txt", ".md"):
            return Path(file_path).read_text(encoding="utf-8")
        elif suffix == ".pdf":
            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    

    #Split text into overlapping chunks.
    def chunk_text(self, text: str) -> List[str]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_ids = input_ids[i:i + self.chunk_size]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        logger.info(f"chunks: {len(chunks)}")
        logger.info(f"list of chunks: {chunks}")
        return chunks
    
    # Add documents to the retriever.
    # Load, chunk, embed, and index documents.
    def add_documents(self, file_paths: List[str]):
        
        all_chunks = []
        for path in file_paths:
            logger.info(f"document path: {file_paths}")
            raw_text = self.load_document(path)
            chunks = self.chunk_text(raw_text)
            all_chunks.extend(chunks)
            logger.info(f"length of total chunks: {len(chunks)}")


        self.documents = all_chunks

        # Create embeddings and build FAISS index
        embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    # Query the indexed documents and return top_k most relevant chunks.
    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        
        if self.index is None:
            raise ValueError("Index not initialized. Add documents first.")
        query_emb = self.model.encode([query_text])
        distances, indices = self.index.search(query_emb, top_k)
        return [self.documents[i] for i in indices[0]]
    
    #Save documents list and FAISS index to disk.
    def save(self, folder: str):
        
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
