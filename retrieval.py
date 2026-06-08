from pathlib import Path
import os
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "uic_unofficial_guide_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TEST_QUERIES = [
    "How difficult is CS 251 according to students?",
    "What advice do students give for succeeding in CS 251?",
    "Which professors are described as approachable or accessible?",
    "What do students say about Natalie Parde?",
]


def ensure_project_environment():
    """
    Re-run this script with the project virtual environment if the current
    Python interpreter does not have the required packages installed.
    """
    try:
        import chromadb  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return
    except ModuleNotFoundError:
        if (
            __name__ == "__main__"
            and VENV_PYTHON.exists()
            and Path(sys.executable).resolve() != VENV_PYTHON.resolve()
        ):
            subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())], check=True)
            sys.exit(0)
        raise


ensure_project_environment()

import chromadb
from sentence_transformers import SentenceTransformer

from pipeline import build_chunks, load_documents


def load_pipeline_chunks():
    """
    Load documents and reuse the existing chunk-building logic from pipeline.py.
    """
    documents = load_documents()
    chunks = build_chunks(documents)
    return documents, chunks


def load_embedding_model():
    """
    Load the embedding model named in planning.md.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_texts(model, texts):
    """
    Encode text with normalized embeddings so cosine-based retrieval is stable.
    The same settings are used for both stored chunks and user queries.
    """
    return model.encode(texts, normalize_embeddings=True).tolist()


def create_fresh_collection():
    """
    Create a fresh ChromaDB collection that uses cosine distance.
    Deleting the old collection avoids duplicate ID errors on reruns.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # This is expected the first time the script runs.
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(collection, model, chunks):
    """
    Store all chunk text, IDs, metadata, and embeddings in ChromaDB.
    """
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]
    embeddings = encode_texts(model, chunk_texts)

    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        metadatas=chunk_metadatas,
        embeddings=embeddings,
    )


def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the top matching chunks for a query.
    Each result includes text, source filename, chunk index, and score values.
    """
    if not hasattr(retrieve, "model"):
        documents, chunks = load_pipeline_chunks()
        model = load_embedding_model()
        collection = create_fresh_collection()
        store_chunks(collection, model, chunks)

        retrieve.documents = documents
        retrieve.chunks = chunks
        retrieve.model = model
        retrieve.collection = collection

    query_embedding = encode_texts(retrieve.model, [query])
    results = retrieve.collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents_list = results.get("documents") or [[]]
    metadatas_list = results.get("metadatas") or [[]]
    distances_list = results.get("distances") or [[]]

    matched_chunks = []
    for document, metadata, distance in zip(
        documents_list[0],
        metadatas_list[0],
        distances_list[0],
    ):
        similarity = 1 - distance
        matched_chunks.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "distance": distance,
                "similarity": similarity,
            }
        )

    return matched_chunks


def print_results_for_query(query, results):
    """
    Print retrieval results in a format that is easy to inspect manually.
    """
    print(f"Query: {query}")
    print("Top returned chunks:")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}:")
        print(f"Source: {result['source']}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Cosine distance (lower is better): {result['distance']:.4f}")
        print(f"Cosine similarity estimate (higher is better): {result['similarity']:.4f}")
        print(f"Text: {result['text']}")
        print("-" * 60)

    print()


def main():
    """
    Build the retrieval system and run the required Milestone 4 test queries.
    """
    results_for_first_query = retrieve(TEST_QUERIES[0], top_k=3)

    print(f"Loaded documents: {len(retrieve.documents)}")
    print(f"Embedded chunks: {len(retrieve.chunks)}")
    print()

    print_results_for_query(TEST_QUERIES[0], results_for_first_query)

    for query in TEST_QUERIES[1:]:
        results = retrieve(query, top_k=3)
        print_results_for_query(query, results)


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    main()
