
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into chunks using character-based windows.

    chunk_size: maximum number of characters in each chunk.
    overlap: number of characters shared between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """
    Split loaded documents into chunks and preserve their metadata.
    """

    all_chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        text_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": index,
                },
            })

    return all_chunks
