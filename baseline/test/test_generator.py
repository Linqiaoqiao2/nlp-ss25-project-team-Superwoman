import unittest
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from retriever.retriever import Retriever
from generator.generator import Generator
from util.fileUtil import FileUtil

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Load the sample file
        current_dir = os.path.dirname(__file__)
        self.document_path = os.path.join(current_dir, "../data/darwins_theory.txt")

        with open(self.document_path, 'r') as f:
            text_content = f.read()


        # Initialize retriever and add the test file
        self.retriever = Retriever()
        self.retriever.add_documents([self.document_path])

        # Load the llm from the folder llms
        model_path = FileUtil.get_model_filePath()
        print(f"filePath: {model_path}")
        self.generator = Generator(filePath=model_path) 

        self.prompt_template = (
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )

    def test_generation(self):
        query = "What was Darwin's contribution?"
        top_chunks = self.retriever.query(query, top_k=2)
        answer = self.generator.generate_answer(query, top_chunks, self.prompt_template)
        print("\nGenerated Answer:\n", answer)

        self.assertIsInstance(answer, str)

if __name__ == "__main__":
    unittest.main()
