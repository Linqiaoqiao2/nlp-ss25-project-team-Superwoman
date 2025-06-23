import re
import os
import sys
from typing import List, Tuple, Union, Any

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langdetect import detect

from util.logger_config import setup_logger

logger = setup_logger("Generator", log_file="logs/RAGPipeline.log")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Generator:
    """
    A wrapper around a local LlamaCpp model that builds a prompt from
    retrieved context and returns an answer. Supports DE/EN language detection.
    """

    def __init__(self, file_path: str, max_tokens= 1024) -> None:
        try:
            logger.info(f"Loading model from: {file_path}")

            self.llm = LlamaCpp(
                model_path=file_path,
                use_mmap=False,
                n_gpu_layers=0,
                n_batch=64,
                n_ctx=max_tokens,  # Must match max_context_tokens
                n_threads=4,
                f16_kv=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False,
            )
        except Exception as e:
            logger.critical(f"Exception while initializing Generator: {e}")
            raise

    @staticmethod
    def clean_query(query: str) -> str:
        return re.sub(' +', ' ', query.strip())

    @staticmethod
    def _extract_text(chunk: Union[str, Tuple[Any, float]]) -> str:
        if isinstance(chunk, tuple):
            raw = chunk[0]
        else:
            raw = chunk

        if hasattr(raw, "page_content"):
            raw = raw.page_content

        return str(raw) if not isinstance(raw, str) else raw

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Union[str, Tuple[Any, float]]],
        prompt_template: str,
        previous_conversation: str = ""
    ) -> str:
        query = self.clean_query(query)
        max_context_tokens = 850
        prompt_chunks: List[str] = []
        token_count = 0
        
        context = "\n".join(prompt_chunks)

        for chunk in context_chunks:
            text = self._extract_text(chunk)
            words = text.split()  # Approximate token count

            if token_count + len(words) > max_context_tokens:
                print(f"\ntotal count of words: {len(token_count)}")
                break

            prompt_chunks.append(text)
            token_count += len(words)

        print(f"\n============context counts: {len(context_chunks)}")

        context = "\n".join(prompt_chunks)
        prompt = prompt_template.format(context=context, query=query)
        print("\n\nprevious_conversation summary:", previous_conversation)
        # Language detection
        try:
            language = detect(query)
            if language == 'de':
                prompt = f"\nBitte beantworte die folgende Frage auf Deutsch: {prompt}"
            elif language == 'en':
                prompt = f"\nPlease answer the following question in English: {prompt}"
            else:
                logger.warning("Unsupported language detected. Defaulting to English.")
                prompt = f"\nPlease answer the following question in English: {prompt}"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return ""
        print(f"\nPrompt =============: {prompt}\n==============")
        return self.llm.invoke(prompt, max_tokens=506)
    

    def summarize_chat_history(self, chat_history, max_tokens=100):
        prompt = f"You are a university chat assitant. Give me a context the following chat:\n\n{chat_history}\n\nSummary:"
        summary = self.llm.invoke(prompt, max_tokens=max_tokens)
        return summary
