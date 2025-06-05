import re
import os
from util.logger_config import setup_logger

from typing import List
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langdetect import detect

logger = setup_logger("Generator", log_file="logs/RAGPipeline.log")

class Generator:
    """
    Generator class that uses a local LlamaCpp model to generate an answer from retrieved context.
    """
    def __init__(self, filePath):
        try:
            logger.info(f"Loading model from: {filePath}")

            current_dir = os.path.dirname(__file__)
            self.llm = LlamaCpp(
                # model_path = os.path.join(current_dir, "orca-mini-3b-gguf2-q4_0.gguf"),
                model_path = filePath,
                n_gpu_layers=0,
                n_batch=64,
                n_ctx=1024,
                f16_kv=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False,
            )
        except Exception as e:
            logger.critical("Exception while calling init Generator: {e}")
            raise

    def clean_query(self, query: str) -> str:
        return re.sub(' +', ' ', query.strip())

    def generate_answer(self, query: str, context_chunks: List[str], prompt_template: str) -> str:
        query = self.clean_query(query)
        
        context = "\n".join(context_chunks)
        prompt = prompt_template.format(context=context, query=query)

        #detect query language
        language = detect(query)
        if language == 'de': 
            prompt = f"Bitte beantworte die folgende Frage auf Deutsch: {prompt}"
        elif language == 'en': 
            prompt = f"Please answer the following question in English: {prompt}"
        else:
            logger.warning("Unsupported language detected. Defaulting to English.")
            prompt = f"Please answer the following question in English: {prompt}"

        return self.llm.invoke(prompt)