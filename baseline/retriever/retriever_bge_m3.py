# pip install -U sentence-transformers transformers rank_bm25 faiss-cpu PyMuPDF
import os, pickle, faiss, fitz
from pathlib import Path
from typing import List, Tuple, Sequence

from sentence_transformers import SentenceTransformer, CrossEncoder #new
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi #new


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
    ):
        self.embedder = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reranker = CrossEncoder(reranker_name)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight

        self.chunks: List[str] = []
        self.bm25: BM25Okapi | None = None
        self.index: faiss.Index | None = None

    #  File loading 
    @staticmethod
    def _load_document(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8")
        if suffix == ".pdf":
            pdf = fitz.open(path)
            return "\n".join(p.get_text() for p in pdf)
        raise ValueError(f"Unsupported format: {suffix}")

    #  Text chunking 
    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        max_model_length = 512 
        if self.chunk_size > max_model_length:
            self.chunk_size = max_model_length

        step = self.chunk_size - self.chunk_overlap
        return [
            self.tokenizer.decode(ids[i : i + self.chunk_size], skip_special_tokens=True)
            for i in range(0, len(ids), step)
        ]

    # Index building 
    def add_documents(self, paths: Sequence[str]) -> None:
        for p in paths:
            self.chunks.extend(self._chunk_text(self._load_document(p)))

        # Dense vectors
        vecs = self.embedder.encode(
            self.chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        )
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        # Sparse BM25
        self.bm25 = BM25Okapi([c.lower().split() for c in self.chunks])

    #  Hybrid search 
    def query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Call add_documents() first.")

        # Dense retrieval
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        d_scores, d_idx = self.index.search(q_vec[None, :], top_k * 4)
        d_idx, d_scores = d_idx[0], d_scores[0]

        # Sparse retrieval
        s_scores = self.bm25.get_scores(query.lower().split())
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

        # Cross-encoder reranking
        pairs = [(query, self.chunks[i]) for i in cand_idx]
        ce_scores = self.reranker.predict(pairs)
        final = sorted(zip(cand_idx, ce_scores), key=lambda x: x[1], reverse=True)[: top_k]

        return [(self.chunks[i], float(s)) for i, s in final]

    #  Persistence 
    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        pickle.dump(self.chunks, open(f"{folder}/chunks.pkl", "wb"))
        pickle.dump(self.bm25, open(f"{folder}/bm25.pkl", "wb"))
        if self.index is not None:
            faiss.write_index(self.index, f"{folder}/dense.faiss")

    def load(self, folder: str) -> None:
        self.chunks = pickle.load(open(f"{folder}/chunks.pkl", "rb"))
        self.bm25 = pickle.load(open(f"{folder}/bm25.pkl", "rb"))
        self.index = faiss.read_index(f"{folder}/dense.faiss")
