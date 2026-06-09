from pathlib import Path
import os
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
OLLAMA_MODEL = "llama3.1:8b"
REFUSAL_MESSAGE = "I don't have enough information in the documents to answer that."
TEST_QUESTIONS = [
    "How difficult is CS 251 according to students?",
    "Which professors are described as approachable or accessible?",
    "What is the best dining hall at UIC?",
]


def ensure_project_environment():
    """
    Re-run this script with the project virtual environment if needed.
    This keeps `python query.py` working even when the system interpreter
    does not have the project packages installed.
    """
    try:
        import ollama  # noqa: F401
        import chromadb  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return
    except ModuleNotFoundError:
        if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
            subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())], check=True)
            sys.exit(0)
        raise


ensure_project_environment()

import ollama

from retrieval import retrieve


def build_context(retrieved_chunks):
    """
    Turn retrieved chunks into a labeled context block for the model.
    Including source names in the context helps the answer stay grounded.
    """
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Source: {chunk['source']}",
                    f"Chunk index: {chunk['chunk_index']}",
                    f"Text: {chunk['text']}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_prompt(question, context):
    """
    Build a grounded prompt that restricts the model to the retrieved context.
    """
    return f"""Answer the question using only the provided context.

Rules:
- Use only the provided context.
- Do not use outside knowledge.
- Do not invent details.
- If the context does not support the answer, say: "{REFUSAL_MESSAGE}"
- Keep the answer concise.
- If you mention sources, use source filenames only.
- Do not mention chunk numbers.

Question:
{question}

Context:
{context}
"""


def unique_sources(retrieved_chunks):
    """
    Return unique source filenames while keeping their original order.
    """
    seen = set()
    sources = []

    for chunk in retrieved_chunks:
        source = chunk["source"]
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources


def append_sources_to_answer(answer_text, sources):
    """
    Append programmatic source attribution directly to the answer text.
    """
    cleaned_answer = answer_text.strip()

    if not sources:
        return f"{REFUSAL_MESSAGE}\n\nSources: No relevant sources found."

    return f"{cleaned_answer}\n\nSources: {', '.join(sources)}"


def ask(question: str) -> dict:
    """
    Retrieve relevant chunks and ask Ollama for a grounded answer.
    If retrieval confidence is weak, refuse before calling the model.
    """
    retrieved_chunks = retrieve(question, top_k=3)
    sources = unique_sources(retrieved_chunks)

    if not retrieved_chunks:
        return {
            "answer": append_sources_to_answer(REFUSAL_MESSAGE, []),
            "sources": [],
            "retrieved_chunks": [],
        }

    best_distance = retrieved_chunks[0]["distance"]
    if best_distance > 0.5:
        return {
            "answer": append_sources_to_answer(REFUSAL_MESSAGE, []),
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
        }

    context = build_context(retrieved_chunks)
    prompt = build_prompt(question, context)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer_text = response["message"]["content"].strip()
    answer_text = append_sources_to_answer(answer_text, sources)

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


def print_result(question, result):
    """
    Print one question, its answer, its sources, and the retrieved chunks.
    """
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print("Sources:")

    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("- None")

    print("Retrieved chunks:")
    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"Result {index}:")
        print(f"Source: {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Distance: {chunk['distance']:.4f}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"Text: {chunk['text']}")
        print("-" * 60)

    print()


def main():
    """
    Run the required Milestone 5 test questions.
    """
    for question in TEST_QUESTIONS:
        result = ask(question)
        print_result(question, result)


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    main()
