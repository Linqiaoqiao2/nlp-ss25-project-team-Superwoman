import os, pickle, faiss, fitz
from pathlib import Path
from typing import List, Tuple, Sequence, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from util.logger_config import setup_logger

logger = setup_logger("Retriever", log_file="logs/RAGPipeline.log")

class Retriever:
    """
    Hybrid dense + sparse retrieval with optional cross-encoder re-ranking.
    Dense encoder  : BAAI/bge-m3
    Sparse retriever: BM25Okapi
    Reranker       : BAAI/bge-reranker-base
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        reranker_name: str = "BAAI/bge-reranker-base",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        dense_weight: float = 0.5,  # 0 → pure BM25, 1 → pure dense
        use_reranker: bool = True
    ):
        # Embedding and tokenizer setup
        self.embedder = SentenceTransformer(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.reranker = CrossEncoder(reranker_name)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder(reranker_name)
        else:
            self.reranker = None
        self.chunks: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.index: Optional[faiss.Index] = None

    # --------- File loading ---------
    @staticmethod
    def _load_document(path: str) -> str:
        """
        Load and extract text from txt/md/pdf/json documents for RAG pipeline.
        """
        suffix = Path(path).suffix.lower()
        if suffix in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8")
        elif suffix == ".pdf":
            import fitz
            pdf = fitz.open(path)
            return "\n".join(p.get_text() for p in pdf)
        elif suffix == ".json":
            import json
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "content" in data:
                return data["content"]
            elif isinstance(data, list):
                return "\n\n".join(item.get("content", "") for item in data if "content" in item)
            else:
                raise ValueError(f"No 'content' field found in JSON: {path}")
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    # --------- Text chunking ---------
    def _chunk_text(self, text: str) -> List[str]:
        """Tokenize text into overlapping chunks."""
        max_model_length = 512
        tokens = self.tokenizer.tokenize(text) #['hello', 'world', '!']
        ids = self.tokenizer.convert_tokens_to_ids(tokens) #	[7592, 2088]

        #  Make sure that chunk_size not exceed model's max length
        chunk_size = min(self.chunk_size, max_model_length)
        step = chunk_size - self.chunk_overlap

        chunks = [
            self.tokenizer.decode(ids[i : i + chunk_size], skip_special_tokens=True)
            for i in range(0, len(ids), step)
        ]
        
        return chunks
    # --------- Tokenization for BM25 ---------
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text using the same tokenizer as dense encoder."""
        return self.tokenizer.tokenize(text.lower())


    # --------- Index building ---------
    def add_documents(self, paths: Sequence[str]) -> None:
        """Build dense FAISS index and sparse BM25 index from input documents."""
        logger.info("Tokenize text into overlapping chunks.")

        for p in paths:
            self.chunks.extend(self._chunk_text(self._load_document(p)))

        # Dense vectors
        vecs = self.embedder.encode(
            self.chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        ) # [num_chunks, embedding_dim]
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        # Sparse BM25
        tokenized_chunks = [self._tokenize_for_bm25(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    # --------- Internal utility: reranker-safe pair truncation ---------
    def _truncate_pair(self, query: str, chunk: str, max_tokens: int = 512) -> Tuple[str, str]:
        """Truncate query + chunk pairs to fit CrossEncoder token limit."""
        q_tokens = self.tokenizer.tokenize(query)
        c_tokens = self.tokenizer.tokenize(chunk)
        remaining = max_tokens - len(q_tokens)
        if remaining < len(c_tokens):
            c_tokens = c_tokens[:remaining]
            chunk = self.tokenizer.convert_tokens_to_string(c_tokens)
        return (query, chunk)

    # --------- Hybrid retrieval ---------
    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        
        """Retrieve top-k relevant text chunks using hybrid dense + sparse + reranker."""
        logger.info("Retrieve top-k relevant text chunks using hybrid dense + sparse + reranker.")
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Call add_documents() first.")

        # Dense retrieval
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        d_scores, d_idx = self.index.search(q_vec[None, :], top_k * 4)
        d_idx, d_scores = d_idx[0], d_scores[0]

        # Sparse retrieval
        query_tokens = self._tokenize_for_bm25(query)
        s_scores = self.bm25.get_scores(query_tokens)
        s_idx = sorted(range(len(s_scores)), key=s_scores.__getitem__, reverse=True)[: top_k * 4]

        # Score fusion
        candidates = set(d_idx) | set(s_idx)
        fused = []
        for i in candidates:
            dense = d_scores[list(d_idx).index(i)] if i in d_idx else 0.0
            sparse = s_scores[i] if i in s_idx else 0.0
            fused.append((i, self.dense_weight * dense + (1 - self.dense_weight) * sparse))
        fused.sort(key=lambda x: x[1], reverse=True)
        cand_idx = [i for i, _ in fused[: top_k * 4]]

        # If reranker is enabled, rerank top candidates
        if hasattr(self, "use_reranker") and self.use_reranker and self.reranker is not None:
            pairs = [self._truncate_pair(query, self.chunks[i]) for i in cand_idx]
            ce_scores = self.reranker.predict(pairs)
            final = sorted(zip(cand_idx, ce_scores), key=lambda x: x[1], reverse=True)[: top_k]
        else:
            # If not using reranker, return top_k from fused scores
            final = fused[:top_k]

        return [(self.chunks[i], float(s)) for i, s in final]

    # --------- Persistence ---------
    def save(self, folder: str) -> None:
        """Save chunks, BM25, and dense index to disk."""
        os.makedirs(folder, exist_ok=True)
        pickle.dump(self.chunks, open(f"{folder}/chunks.pkl", "wb"))
        pickle.dump(self.bm25, open(f"{folder}/bm25.pkl", "wb"))
        if self.index is not None:
            faiss.write_index(self.index, f"{folder}/dense.faiss")

    def load(self, folder: str) -> None:
        """Load chunks, BM25, and dense index from disk."""
        self.chunks = pickle.load(open(f"{folder}/chunks.pkl", "rb"))
        self.bm25 = pickle.load(open(f"{folder}/bm25.pkl", "rb"))
        self.index = faiss.read_index(f"{folder}/dense.faiss")
