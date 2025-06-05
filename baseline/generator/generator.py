import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Tuple, Union, Any

from util.logger_config import setup_logger
from util.fileUtil import FileUtil  # Kept for future use
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler

logger = setup_logger("Generator", log_file="logs/RAGPipeline.log")


class Generator:
    """
    A wrapper around a local LlamaCpp model that builds a prompt from
    retrieved context and returns an answer.
    """

    def __init__(self, file_path: str) -> None:
        try:
            logger.info(f"Loading model from: {file_path}")

            self.llm = LlamaCpp(
                model_path=file_path,
                use_mmap=False,
                n_gpu_layers=0,
                n_batch=64,
                n_ctx=1024,        # Keep in sync with `max_context_tokens`
                n_threads=4,
                f16_kv=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False,
            )
        except Exception as e:
            logger.critical(f"Exception while initializing Generator: {e}")
            raise

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _extract_text(chunk: Union[str, Tuple[Any, float]]) -> str:
        """
        Normalize a chunk to plain text.

        Accepted formats:
        - str
        - (str, score)
        - (Document, score)

        Parameters
        ----------
        chunk : str | tuple
            The context chunk to normalize.

        Returns
        -------
        str
            Plain text extracted from the chunk.
        """
        # Tuple -> take element 0
        if isinstance(chunk, tuple):
            raw = chunk[0]
        else:
            raw = chunk

        # LangChain Document -> take `page_content`
        if hasattr(raw, "page_content"):
            raw = raw.page_content

        # Fallback to str
        if not isinstance(raw, str):
            raw = str(raw)

        return raw

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Union[str, Tuple[Any, float]]],
        prompt_template: str,
    ) -> str:
        """
        Build a prompt from `query` and `context_chunks`, then invoke the model.

        Parameters
        ----------
        query : str
            The user question.
        context_chunks : list
            Retrieved passages (str) or (passage, score) tuples.
        prompt_template : str
            Template string with placeholders `{context}` and `{query}`.

        Returns
        -------
        str
            The generated answer from LlamaCpp.
        """
        max_context_tokens = 1024  # Must match `n_ctx` above
        prompt_chunks: List[str] = []
        token_count = 0

        for chunk in context_chunks:
            text = self._extract_text(chunk)
            words = text.split()  # Simple token approximation

            # Truncate to avoid exceeding model context
            if token_count + len(words) > max_context_tokens:
                break

            prompt_chunks.append(text)
            token_count += len(words)

        context = "\n".join(prompt_chunks)
        prompt = prompt_template.format(context=context, query=query)

        logger.info(f"[Generator] Prompt approx. length: {token_count} tokens")

        # Invoke the local model and return its response
        return self.llm.invoke(prompt)
