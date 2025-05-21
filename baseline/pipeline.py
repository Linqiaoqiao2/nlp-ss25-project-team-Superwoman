# pipeline.py
from retriever.retriever import Retriever
from generator.generator import Generator
from util.fileUtil import FileUtil

class RAGPipeline:
    def __init__(self, document_paths: list[str], prompt_template: str):
        self.retriever = Retriever()
        self.retriever.add_documents(document_paths)

        model_path = FileUtil.get_model_filePath()
        self.generator = Generator(filePath=model_path)
        self.prompt_template = prompt_template

    def run(self, query: str, top_k: int = 5) -> str:
        top_chunks = self.retriever.query(query, top_k=top_k)
        return self.generator.generate_answer(query, top_chunks, self.prompt_template)
