import os
import sys
import unittest
from pathlib import Path

# Make project root importable
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from retriever.retriever_bge_m3 import Retriever
from pipeline import RAGPipeline


class TestGenerator(unittest.TestCase):
    """
    Unit-test suite for the RAGPipeline.
    It indexes every file inside data/cleaned_* and checks that answers are strings.
    """

    @classmethod
    def setUpClass(cls):
        # Use your actual data directory
        data_dir = Path("/Users/subhasrisuresh/mastersProject/git_project/nlp-ss25-project-team-Superwoman/baseline/data")

        # Find all subdirectories starting with 'cleaned_'
        cleaned_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cleaned_")]
        if not cleaned_dirs:
            raise FileNotFoundError(f"No folder starting with 'cleaned_' found in {data_dir}")
        
        cleaned_dir = cleaned_dirs[0]

        allowed_suffixes = {".txt", ".md", ".pdf"}
        document_paths = [
            f.as_posix()
            for f in cleaned_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]

        cls.pipeline = RAGPipeline(
            document_paths=document_paths,
            prompt_template=("You are a helpful assistant for question-answering tasks. Take the following question (the user query) and use this helpful information (the data retrieved in the similarity search) to answer it. If you don't know the answer based on the information provided, just say you don't know."
                "Context:\n{context}\n\n"
                "Question:\n{query}\n\n"
                "Answer:"
            )
        )

        base_dir = Path(__file__).resolve().parent
        cls.output_file = base_dir / "outputs" / "answers.txt"
        cls.output_file.parent.mkdir(exist_ok=True)
        cls.output_file.write_text("=== Summarization Test Output ===\n\n", encoding="utf-8")

    @classmethod
    def _write_output(cls, query: str, answer: str) -> None:
        with cls.output_file.open("a", encoding="utf-8") as f:
            f.write(f"Query: {query.strip()}\n")
            f.write(f"Answer: {answer.strip()}\n")
            f.write("-" * 40 + "\n")

    def test_ask_questions(self):
        """Iterate through questions.txt and ensure answers are strings."""
        questions_file = Path(__file__).resolve().parent / "input" / "questions.txt"

        with questions_file.open("r", encoding="utf-8") as qf:
            for query in qf:
                # answer = self.pipeline.run(query)
                answer = self.pipeline.run_chatbot(query)

                print("Q:", query.strip())
                print("A:", answer.strip())
                self._write_output(query, answer)
                self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()
