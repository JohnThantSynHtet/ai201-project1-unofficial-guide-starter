from pathlib import Path
import re
import random


DOCUMENTS_DIR = Path("documents")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MIN_FINAL_CHUNK_SIZE = 100
MIN_STANDALONE_CHUNK_SIZE = 175
MAX_CHUNK_OVERSHOOT = 75


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


def split_sentences(text):
    """
    Split text into readable sentences.
    This keeps chunk boundaries aligned with full thoughts instead of raw characters.
    """
    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []

    for part in sentence_parts:
        sentence = part.strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def join_sentences(sentences):
    """
    Join a list of sentences back into one chunk of text.
    """
    return " ".join(sentences).strip()


def choose_overlap_sentences(chunk_sentences, overlap_size):
    """
    Choose how many ending sentences to reuse in the next chunk.
    The overlap stays sentence-based, so no words or sentences get cut in half.
    """
    if len(chunk_sentences) <= 1:
        return 0

    overlap_char_count = 0
    overlap_sentence_count = 0
    max_overlap_sentences = len(chunk_sentences) - 1

    for sentence in reversed(chunk_sentences):
        if overlap_sentence_count >= max_overlap_sentences:
            break

        if overlap_sentence_count == 0:
            overlap_char_count += len(sentence)
        else:
            overlap_char_count += len(sentence) + 1

        overlap_sentence_count += 1

        if overlap_char_count >= overlap_size:
            break

    return overlap_sentence_count


def build_sentence_chunk_groups(
    sentences,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Build chunks by combining full sentences until each chunk is near the target size.
    """
    chunk_groups = []
    start_index = 0

    while start_index < len(sentences):
        current_sentences = []
        current_length = 0
        index = start_index

        while index < len(sentences):
            sentence = sentences[index]
            added_length = len(sentence) if not current_sentences else len(sentence) + 1
            candidate_length = current_length + added_length

            if current_sentences and candidate_length > chunk_size:
                current_distance = abs(chunk_size - current_length)
                candidate_distance = abs(chunk_size - candidate_length)

                if (
                    candidate_length > chunk_size + MAX_CHUNK_OVERSHOOT
                    or candidate_distance > current_distance
                ):
                    break

            current_sentences.append(sentence)
            current_length = candidate_length
            index += 1

            if current_length >= chunk_size:
                break

        if not current_sentences:
            break

        chunk_groups.append(current_sentences)

        if index >= len(sentences):
            break

        overlap_sentence_count = choose_overlap_sentences(current_sentences, overlap)
        start_index = index - overlap_sentence_count

    return chunk_groups


def find_sentence_overlap(previous_chunk, current_chunk):
    """
    Find repeated boundary sentences shared by two neighboring chunks.
    This helps merge a short final chunk without duplicating overlap text.
    """
    max_possible_overlap = min(len(previous_chunk), len(current_chunk))

    for overlap_size in range(max_possible_overlap, 0, -1):
        if previous_chunk[-overlap_size:] == current_chunk[:overlap_size]:
            return overlap_size

    return 0


def merge_short_final_chunk(chunk_groups, min_final_chunk_size=MIN_FINAL_CHUNK_SIZE):
    """
    Merge a very short leftover chunk into the previous chunk.
    This avoids tiny or weak tail chunks at the end of a document.
    """
    if len(chunk_groups) < 2:
        return chunk_groups

    final_chunk_text = join_sentences(chunk_groups[-1])
    merge_threshold = max(min_final_chunk_size, MIN_STANDALONE_CHUNK_SIZE)

    if len(final_chunk_text) >= merge_threshold:
        return chunk_groups

    previous_chunk = chunk_groups[-2]
    final_chunk = chunk_groups[-1]
    overlap_size = find_sentence_overlap(previous_chunk, final_chunk)

    chunk_groups[-2] = previous_chunk + final_chunk[overlap_size:]
    chunk_groups.pop()

    return chunk_groups


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into sentence-aware chunks with sentence-based overlap.
    """
    sentences = split_sentences(text)
    chunk_groups = build_sentence_chunk_groups(sentences, chunk_size, overlap)
    chunk_groups = merge_short_final_chunk(chunk_groups)

    chunks = []
    for sentence_group in chunk_groups:
        chunk = join_sentences(sentence_group)
        if chunk:
            chunks.append(chunk)

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
