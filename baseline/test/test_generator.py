import unittest
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from retriever.retriever import Retriever
from generator.generator import Generator

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary text file with sample content
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, "darwins_theory.txt")

        # Write sample content to the file
        sample_text = (
            "Charles Darwin introduced the theory of evolution by natural selection in his book "
            "'On the Origin of Species'. This theory proposes that species evolve over generations "
            "through a process of natural selection."
        )
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(sample_text)

        # Initialize retriever and add the test file
        self.retriever = Retriever()
        self.retriever.add_documents([self.test_file_path])

        # Note: Provide actual path to a local LLaMA model if testing generation
        self.generator = Generator(model_path="path/to/llama/model.gguf")  # Replace with real path

        self.prompt_template = (
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )

    def test_generation(self):
        query = "What was Darwin's contribution to evolutionary biology?"
        top_chunks = self.retriever.query(query, top_k=2)
        answer = self.generator.generate_answer(query, top_chunks, self.prompt_template)

        self.assertIsInstance(answer, str)
        print("\nGenerated Answer:\n", answer)

    def tearDown(self):
        self.temp_dir.cleanup()

if __name__ == "__main__":
    unittest.main()
