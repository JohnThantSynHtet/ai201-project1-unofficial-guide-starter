from pathlib import Path
import os
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"


def ensure_project_environment():
    """
    Re-run this script with the project virtual environment if Gradio is missing.
    """
    try:
        import gradio  # noqa: F401
        return
    except ModuleNotFoundError:
        if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
            subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())], check=True)
            sys.exit(0)
        raise


ensure_project_environment()

import gradio as gr

from query import ask


def format_sources(sources):
    """
    Format source filenames as bullet points for display in the UI.
    """
    if not sources:
        return "- None"

    return "\n".join(f"- {source}" for source in sources)


def format_retrieved_chunks(retrieved_chunks):
    """
    Format the retrieved chunks so they are easy to inspect during demos.
    """
    if not retrieved_chunks:
        return "No chunks retrieved."

    chunk_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        chunk_blocks.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"Source: {chunk['source']}",
                    f"Chunk index: {chunk['chunk_index']}",
                    f"Distance: {chunk['distance']:.4f}",
                    f"Similarity: {chunk['similarity']:.4f}",
                    f"Text: {chunk['text']}",
                ]
            )
        )

    return "\n\n".join(chunk_blocks)


def answer_question(question):
    """
    Run the grounded QA pipeline and return UI-friendly strings.
    """
    result = ask(question)
    return (
        result["answer"],
        format_sources(result["sources"]),
        format_retrieved_chunks(result["retrieved_chunks"]),
    )


with gr.Blocks() as demo:
    gr.Markdown("# The Unofficial Guide")

    question_input = gr.Textbox(label="Your question", lines=2)
    ask_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer", lines=5)
    sources_output = gr.Textbox(label="Retrieved from", lines=5)
    chunks_output = gr.Textbox(label="Retrieved chunks", lines=14)

    ask_button.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output, chunks_output],
    )


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    print("Starting Gradio UI...")
    demo.launch()
