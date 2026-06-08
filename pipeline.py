from pathlib import Path
import re
import random


DOCUMENTS_DIR = Path("documents")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def clean_text(text):
    """
    Clean raw document text.
    Keeps useful review content but removes extra spacing.
    """
    text = text.replace("&amp;", "&")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#39;", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents():
    """
    Load all .txt files from the documents folder.
    Returns a list of dictionaries with source and text.
    """
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            documents.append({
                "source": file_path.name,
                "text": cleaned_text
            })

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 0:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(documents):
    """
    Create chunks from all loaded documents.
    Keeps source metadata attached to each chunk.
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['source']}_chunk_{index}",
                "source": doc["source"],
                "chunk_index": index,
                "text": chunk
            })

    return all_chunks


def main():
    documents = load_documents()
    chunks = build_chunks(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print()

    print("Five sample chunks:")
    print("-------------------")

    sample_chunks = random.sample(chunks, min(5, len(chunks)))

    for chunk in sample_chunks:
        print(f"ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text']}")
        print("-------------------")


if __name__ == "__main__":
    main()