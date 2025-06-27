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

    def run_chatbot(self, query: str, top_k: int = 3, previous_conversation: str = "") -> str:
        while (True):
            # query = input("\nPlease type your question: ")
            top_chunks = self.retriever.query(query, top_k=2)
            logger.info(f"Top chunks choosen\n: {top_chunks}")
            result = self.generator.generate_answer(query, top_chunks, self.prompt_template, previous_conversation)
            # previous_conversation += query + result
            logger.info(f"Question: {query}")
            logger.info(f"Answer: {result}")
            # print(f"? conversation history: {previous_conversation}")
            print("\nEnd of answer\n\n\n")
            return result
    
    def getChatSummary(self, chat: str):
        return self.generator.summarize_chat_history(chat)
        
