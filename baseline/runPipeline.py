from pathlib import Path
import gradio as gr
from util.fileUtil import FileUtil
from pipeline import RAGPipeline
from generator.generator import Generator


def main():

    document_paths = FileUtil.get_all_cleaned_data_filenames()

    # 2. Instantiate RAGPipeline
    prompt_template=("You are a helpful assistant for question-answering tasks regarding the university course details. Take the following question (the user query) and use this helpful information (the data retrieved in the similarity search) to answer it. If you don't know the answer based on the information provided, just say you don't know."
        "Context:\n{context}\n\n"
        "Question:\n{query}\n\n"
        "Answer:"
    )
    pipeline = RAGPipeline(document_paths=document_paths, prompt_template=prompt_template)

    # Define a wrapper for the chatbot interaction
    def chatbot_response(message, history, summary=""):
        response = pipeline.run_chatbot(message)  # your existing logic
        history.append((message, response))
        new_qa = f"User: {message}\nBot: {response}"
        summary = pipeline.getChatSummary(summary+new_qa)
        print(f"? conversation summary: {summary}")
        return history

    # Build Gradio chatbot interface
    with gr.Blocks() as chat:
        gr.Markdown("# 🎓 University Bot")
        gr.Markdown("Ask a question and get an answer based on university course and site information.")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter your question and press Enter")
        # clear = gr.Button("Clear Chat")

        msg.submit(chatbot_response, [msg, chatbot], chatbot)
        # clear.click(lambda: [], None, chatbot)

    chat.launch()

if __name__ == "__main__":
    main()
