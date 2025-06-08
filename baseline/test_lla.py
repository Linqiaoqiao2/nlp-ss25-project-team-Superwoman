from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler

llm = LlamaCpp(
    model_path="D:/Model_bge_m3/Model_bge_m3/baseline/llms/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  
    n_ctx=1024,
    n_gpu_layers=0,        
    n_batch=64,
    f16_kv=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True,
)
response = llm.invoke("����һ�仰�������Լ���")
print(response)
