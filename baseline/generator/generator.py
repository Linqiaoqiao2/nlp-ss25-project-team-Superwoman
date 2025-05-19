from typing import List
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

class Generator:
    """
    Generator class that uses a local LlamaCpp model to generate an answer from retrieved context.
    """
    def __init__(self, model_path: str):
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=0,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=False,
        )

    def generate_answer(self, query: str, context_chunks: List[str], prompt_template: str) -> str:
       
        context = "\n".join(context_chunks)
        prompt = prompt_template.format(context=context, query=query)
        return self.llm(prompt)
