import sys
import os

from typing import List
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler

class Generator:
    """
    Generator class that uses a local LlamaCpp model to generate an answer from retrieved context.
    """
    def __init__(self, filePath):
        try:
            print(f"? Loading model from: {filePath}")
            current_dir = os.path.dirname(__file__)
            self.llm = LlamaCpp(
                # model_path = os.path.join(current_dir, "orca-mini-3b-gguf2-q4_0.gguf"),
                model_path = filePath,
                n_gpu_layers=0,
                n_batch=32,
                n_ctx=512,
                f16_kv=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=False,
            )
        except Exception as e:
            print(f"? Failed to load model: {e}")
            raise

    def generate_answer(self, query: str, context_chunks: List[str], prompt_template: str) -> str:
        context = "\n".join(context_chunks)
        prompt = prompt_template.format(context=context, query=query)
        return self.llm.invoke(prompt)