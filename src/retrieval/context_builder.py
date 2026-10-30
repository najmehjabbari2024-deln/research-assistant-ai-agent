
def build_context(retrieved_chunks, max_context_chars=6000):
    """
    Build a structured evidence context for the LLM.

    Each evidence item receives a unique ID and retains
    its source document, page number, and relevance score.
    """

    if not retrieved_chunks:
        return "No relevant information was retrieved."

    context_parts = []
    current_length = 0

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        evidence_id = f"E{rank}"

        source = chunk["source"]
        page = chunk["page"]
        score = chunk["score"]
        content = chunk["text"]

        evidence_block = (
            f"[EVIDENCE_ID: {evidence_id}]\n"
            f"SOURCE_DOCUMENT: {source}\n"
            f"SOURCE_PAGE: {page}\n"
            f"RELEVANCE_SCORE: {score:.4f}\n"
            f"EVIDENCE_TEXT:\n"
            f"{content}\n"
            f"{'=' * 60}\n"
        )

        # Do not exceed the maximum context length
        if current_length + len(evidence_block) > max_context_chars:
            break

        context_parts.append(evidence_block)
        current_length += len(evidence_block)

    return "\n".join(context_parts)
