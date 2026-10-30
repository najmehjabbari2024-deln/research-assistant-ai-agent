import re


def chunk_pages(pages, chunk_size=1000, overlap=200):
    """
    Split page text into sentence-aware chunks
    while preserving source and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        # Step 1: Split text into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        current_sentences = []
        current_length = 0

        # Step 2: Build chunks using complete sentences
        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if (
                current_length + len(sentence) > chunk_size
                and current_sentences
            ):

                chunk_text = " ".join(current_sentences)

                chunks.append({
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_id": len(chunks) + 1,
                    "text": chunk_text
                })

                # Step 3: Preserve some previous sentences as overlap
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(current_sentences):

                    if overlap_length + len(previous_sentence) > overlap:
                        break

                    overlap_sentences.insert(0, previous_sentence)
                    overlap_length += len(previous_sentence)

                current_sentences = overlap_sentences
                current_length = overlap_length

            current_sentences.append(sentence)
            current_length += len(sentence)

        # Step 4: Save the remaining sentences
        if current_sentences:

            chunk_text = " ".join(current_sentences)

            chunks.append({
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_id": len(chunks) + 1,
                "text": chunk_text
            })

    return chunks