from typing import List
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler

class Generator:
    """
    Generator class that uses a local LlamaCpp model to generate an answer from retrieved context.
    """
    def __init__(self, model_path: str):
        print(f"? Loading model from: {model_path}")
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=0,
                n_batch=512,
                n_ctx=2048,
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
