import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import RAGPipeline

class TestGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        current_dir = os.path.dirname(__file__)
        document_path = os.path.join(current_dir, "../data/darwins_theory.txt")
        self.pipeline = RAGPipeline(
            document_paths=[document_path],
            prompt_template=(
                "Context:\n{context}\n\n"
                "Question:\n{query}\n\n"
                "Answer:"
            )
        )
        self.output_file = os.path.join(os.path.dirname(__file__), "outputs", "answers.txt")

        # Overwrite (clear) file at the start of test suite
        with open(self.output_file, "w") as f:
            f.write("=== Summarization Test Output ===\n\n")
    
    def writeOutput(self, query, answer):
        # os.makedirs(os.path.dirname(output_file), exist_ok=True) 
        with open(self.output_file, "a") as f:  # Append mode to keep adding answers
            f.write(f"Query: {query}\n")
            f.write(f"Answer: {answer}\n")
            f.write("-" * 40 + "\n")  


    def test_askQuestions(self):
        self.questions_file = os.path.join(os.path.dirname(__file__), "input", "questions.txt")

        with open(self.questions_file, "r", encoding="utf-8") as file:
            for query in file:
                answer = self.pipeline.run(query)
                self.writeOutput(query, answer)
                self.assertIsInstance(answer, str)

if __name__ == "__main__":
    unittest.main()
