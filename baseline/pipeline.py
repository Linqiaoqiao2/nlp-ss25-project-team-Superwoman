# pipeline.py
from retriever.retriever_bge_m3 import Retriever
from generator.generator import Generator
from util.fileUtil import FileUtil
from util.logger_config import setup_logger

logger = setup_logger("pipeline", log_file="logs/RAGPipeline.log")


class RAGPipeline:
    def __init__(self, document_paths: list[str], prompt_template: str):
        self.retriever = Retriever()
        self.retriever.add_documents(document_paths)

        model_path = FileUtil.get_model_filePath()
        self.generator = Generator(file_path=model_path)
        self.prompt_template = prompt_template

    def run(self, query: str, top_k: int = 3) -> str:
        top_chunks = self.retriever.query(query, top_k=top_k)
        logger.info(f"Top chunks choosen:\n")
        logger.info(f"------>\n".join(top_chunks))
        return self.generator.generate_answer(query, top_chunks, self.prompt_template)
    
    def run_chatbot(self, top_k: int = 3) -> str:
        previous_conversation = "Conversation history:\n"
        while (True):
            query = input("\nPlease type your question: ")
            top_chunks = self.retriever.query(query, top_k=top_k)
            logger.info(f"Top chunks choosen: {top_chunks}")
            result = self.generator.generate_answer(query, top_chunks, self.prompt_template, previous_conversation)
            previous_conversation += query + result
            print("\nEnd of answer")
