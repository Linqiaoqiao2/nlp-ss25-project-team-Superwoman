import os
import sys
import json
from pathlib import Path
from collections import Counter
# Make project root importable
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from util.logger_config import setup_logger
from pipeline import RAGPipeline
from util.fileUtil import FileUtil
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

logger = setup_logger("F1_score", log_file="logs/RAGPipeline.log")

class TestRagOutput:

    @classmethod
    def setUpClass(cls):  
        print("Calculating Bert score...")

        current_dir = Path(__file__).parent
        data_dir = (current_dir / "../data").resolve()

        # Find all subdirectories starting with 'cleaned_'
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

        cls.pipeline = RAGPipeline(
            document_paths=document_paths,
            prompt_template=(
                "You are a helpful assistant for question-answering tasks. Take the following question (the user query) and use this helpful information (the data retrieved in the similarity search) to answer it. "
                "If you don't know the answer based on the information provided, just say you don't know.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{query}\n\n"
                "Answer:"
            )
        )

        cls.output_file = current_dir / "outputs" / "bert_output.txt"
        cls.output_file.parent.mkdir(exist_ok=True)
        cls.output_file.write_text("=== Summarization Bert score Output ===\n\n", encoding="utf-8")
        
        cls.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls.model = BertModel.from_pretrained('bert-base-uncased')


    @classmethod
    def _write_output(cls, query: str, answer: str, generatedAns: str, score: float) -> None:
        with cls.output_file.open("a", encoding="utf-8") as f:
            f.write(f"***Query: {query.strip()}\n")
            f.write(f"***Answer: {answer.strip()}\n")
            f.write(f"***Generated Answer: {generatedAns.strip()}\n")
            f.write(f"***Bert Score: {score:.4f}\n")
            f.write("-" * 40 + "\n")

    @classmethod
    def get_embeddings(cls, inputs):
        with torch.no_grad():
            outputs = cls.model(**inputs)
            # Get the embeddings of the [CLS] token
        return outputs.last_hidden_state[:, 0, :]

    @classmethod
    def compute_answer_Bert(cls):
        cls.setUpClass()
        input_path = Path(__file__).parent / "input" / "bert_qa_input.json"

        with input_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            question = item['qn']
            ground_truth = item['ans']

            # Optional: change 'summary' to a meaningful value if needed
            answer = cls.pipeline.run_rag(question, previous_conversation=None)
            print("-" * 40)

            inputs = cls.tokenizer([ground_truth, answer], padding=True, truncation=True, return_tensors='pt')
            embeddings = cls.get_embeddings(inputs)
            similarity = cosine_similarity(embeddings.numpy())
            bert = similarity[0][1] 

            cls._write_output(question,ground_truth, answer, bert)


if __name__ == "__main__":
    TestRagOutput.compute_answer_Bert()