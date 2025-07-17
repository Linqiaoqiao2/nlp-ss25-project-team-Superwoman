import os
import sys

from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import RAGPipeline
from pathlib import Path

embedder = SentenceTransformer("BAAI/bge-m3")

base_dir = Path(__file__).resolve().parent.parent
questions_file = base_dir / "input" / "questions.txt"
output_file = base_dir / "outputs" / "answers.txt"


data_dir = Path("D:/Model_bge_m3/Model_bge_m3/baseline/data")
cleaned_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cleaned_")]
if not cleaned_dirs:
    raise FileNotFoundError(f"No folder starting with 'cleaned_' found in {data_dir}")
cleaned_dir = cleaned_dirs[0]
allowed_suffixes = {".txt", ".md", ".pdf", ".json"}

document_paths = [
    f.as_posix()
    for f in cleaned_dir.rglob("*")
    if f.is_file() and f.suffix.lower() in allowed_suffixes
]


pipeline = RAGPipeline(
    document_paths=document_paths,
    prompt_template="Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
)

def compute_query_chunk_similarity(query, chunks):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    chunk_embs = embedder.encode(chunks, convert_to_tensor=True)
    return util.cos_sim(query_emb, chunk_embs).mean().item()

def compute_answer_context_similarity(answer, context):
    answer_emb = embedder.encode(answer, convert_to_tensor=True)
    context_emb = embedder.encode(context, convert_to_tensor=True)
    return util.cos_sim(answer_emb, context_emb).item()


with open(questions_file, "r", encoding="utf-8") as qf:
    questions = [q.strip() for q in qf.readlines() if q.strip()]

print(f"? Evaluating {len(questions)} questions...\n")
output_file.parent.mkdir(exist_ok=True)
output_file.write_text("=== Unsupervised Evaluation Results ===\n\n", encoding="utf-8")

for query in tqdm(questions):
    top_chunks = pipeline.retriever.query(query, top_k=3)
    answer = pipeline.generator.generate_answer(query, top_chunks, pipeline.prompt_template)

    sim_q_chunk = compute_query_chunk_similarity(query, top_chunks)
    sim_a_ctx = compute_answer_context_similarity(answer, "\n".join(top_chunks))

    with output_file.open("a", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Answer: {answer.strip()}\n")
        f.write(f"Query ? Chunk Avg Similarity: {sim_q_chunk:.4f}\n")
        f.write(f"Answer ? Context Similarity: {sim_a_ctx:.4f}\n")
        f.write("-" * 40 + "\n")

print("? Evaluation finished. Results saved to:", output_file)
 