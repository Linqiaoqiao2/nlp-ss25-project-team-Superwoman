import os, pickle, faiss, fitz
from pathlib import Path
from typing import List, Tuple, Sequence, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from util.logger_config import setup_logger
from util.fileUtil import FileUtil


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
        chunk_size: int = 64,
        chunk_overlap: int = 16,
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
        Robustly handles different structures in JSON and alerts on empty content.
        """
        suffix = Path(path).suffix.lower()

        try:
            if suffix in (".txt", ".md"):
                text = Path(path).read_text(encoding="utf-8").strip()
                if not text:
                    print(f"⚠️ WARNING: File {path} is empty.")
                return text

            elif suffix == ".pdf":
                import fitz
                pdf = fitz.open(path)
                text = "\n".join(page.get_text() for page in pdf).strip()
                if not text:
                    print(f"⚠️ WARNING: PDF {path} contains no extractable text.")
                return text

            elif suffix == ".json":
                import json
                data = json.loads(Path(path).read_text(encoding="utf-8"))

                # If dict, try common content fields
                if isinstance(data, dict):
                    for key in ["content", "text", "body"]:
                        if key in data and isinstance(data[key], str) and data[key].strip():
                            return data[key].strip()
                    # If dict but no matching field, fallback to JSON string
                    print(f"⚠️ WARNING: No usable text field found in {path}, returning raw JSON string.")
                    return json.dumps(data, ensure_ascii=False)

                # If list of dicts, concatenate valid 'content' fields
                elif isinstance(data, list):
                    contents = [
                        item.get("content", "").strip()
                        for item in data
                        if isinstance(item, dict) and "content" in item and item.get("content", "").strip()
                    ]
                    if contents:
                        return "\n\n".join(contents)
                    else:
                        print(f"⚠️ WARNING: List JSON {path} contains no valid 'content' fields, returning raw JSON string.")
                        return json.dumps(data, ensure_ascii=False)

                else:
                    raise ValueError(f"Unsupported JSON structure in {path}")

            else:
                raise ValueError(f"Unsupported file format for path: {path}")

        except Exception as e:
            print(f"❌ ERROR while loading {path}: {e}")
            return ""


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
        """
        Build dense FAISS index and sparse BM25 index from input documents.
        Tracks source URLs for future citation display in RAG responses.
        """

        logger.info("Tokenizing text into overlapping chunks.")

        # Initialize chunks and their corresponding source URLs
        self.chunks: List[str] = []
        self.chunk_sources: List[str] = []

        for p in paths:
            text = self._load_document(p)
            chunks = self._chunk_text(text)
            self.chunks.extend(chunks)

            # Track source URL or filename for each chunk for traceability
            url = FileUtil.get_url_from_path(p) if hasattr(FileUtil, "get_url_from_path") else str(Path(p).name)
            self.chunk_sources.extend([url] * len(chunks))

        # Debug prints for inspection
        print(f"❓ Debug: Total chunks prepared: {len(self.chunks)}")
        for idx, chunk in enumerate(self.chunks[:3]):
            print(f"Chunk[{idx}] len={len(chunk)}: {repr(chunk[:100])}")
        print(f"❓ Debug: Corresponding sources: {self.chunk_sources[:3]}")

        # Safety check to prevent empty index creation
        if not self.chunks:
            raise ValueError("❌ No valid chunks found. Check your data files or chunking logic.")

        # Create dense embeddings using the embedder
        vecs = self.embedder.encode(
            self.chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )  # shape: [num_chunks, embedding_dim]

        print(f"✅ Debug: vecs.shape after encoding: {vecs.shape}")

        # Check for empty embeddings to prevent IndexError
        if vecs.shape[0] == 0:
            raise ValueError("❌ Embedding returned empty vectors. Check model availability or input data quality.")

        # Build FAISS dense index
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        # Build BM25 sparse index for hybrid retrieval
        tokenized_chunks = [self._tokenize_for_bm25(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        logger.info(f"✅ Finished building dense and sparse indexes with {len(self.chunks)} chunks.")

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

        return [(self.chunks[i], self.chunk_sources[i], float(s)) for i, s in final]


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
