from pathlib import Path
import gradio as gr
from util.fileUtil import FileUtil
from pipeline import RAGPipeline
from generator.generator import Generator


def main():
    # data_dir = Path("D:/Model_bge_m3/Model_bge_m3/baseline/data/cleaned_json")

    current_dir = Path(__file__).parent
    data_dir = (current_dir / "data").resolve()
    print(f"Data directory path: {data_dir}")

    # Find all 'cleaned_' directories
    cleaned_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("cleaned_")]
    print(f"cleaned directory path: {cleaned_dirs}")
    if not cleaned_dirs:
        raise FileNotFoundError(f"No folder starting with 'cleaned_' found in {data_dir}")


    cleaned_dir = cleaned_dirs[0]

    allowed_suffixes = {".txt", ".md", ".pdf", ".json"}
    document_paths = [
        f.as_posix()
        for f in cleaned_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in allowed_suffixes
    ]

    # 2. Instantiate RAGPipeline
    prompt_template=("You are a helpful assistant for question-answering tasks regarding the university course details. Take the following question (the user query) and use this helpful information (the data retrieved in the similarity search) to answer it. If you don't know the answer based on the information provided, just say you don't know."
        "Context:\n{context}\n\n"
        "Question:\n{query}\n\n"
        "Answer:"
    )
    pipeline = RAGPipeline(document_paths=document_paths, prompt_template=prompt_template)

    # Define a wrapper for the chatbot interaction
    def chatbot_response(message, history, summary=""):
        response = pipeline.run_rag(message)
        history.append((message, response))
        new_qa = f"User: {message}\nBot: {response}"
        summary = pipeline.getChatSummary(summary+new_qa)
        print(f"? conversation summary: {summary}")
        return history, ""

    # Build Gradio chatbot interface
    with gr.Blocks() as chat:
        gr.Markdown("# 🎓 University Bot")
        gr.Markdown("Ask a question and get an answer based on university course and site information.")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter your question and press Enter")
        # clear = gr.Button("Clear Chat")

        msg.submit(chatbot_response, [msg, chatbot],  [chatbot, msg])
        # clear.click(lambda: [], None, chatbot)

    chat.launch()

if __name__ == "__main__":
    main()
