
from pathlib import Path

from src.document_processing.pdf_processor import extract_text_from_pdf
from src.document_processing.text_chunker import chunk_pages
from src.retrieval.retriever import TfidfRetriever
from src.llm.ollama_client import OllamaClient
from src.agents.research_agent import ResearchAgent
from src.workflow.research_workflow import ResearchWorkflow


def main():

    # Step 1: Load academic PDF documents
    data_folder = Path("data")
    pdf_files = sorted(data_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return

    all_chunks = []

    # Step 2: Extract text and create chunks
    for pdf_file in pdf_files:

        pages = extract_text_from_pdf(pdf_file)

        chunks = chunk_pages(
            pages,
            chunk_size=1000,
            overlap=200
        )

        all_chunks.extend(chunks)

        print(
            f"Processed: {pdf_file.name} | "
            f"Pages: {len(pages)} | "
            f"Chunks: {len(chunks)}"
        )

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 3: Build the document retrieval index
    retriever = TfidfRetriever()

    indexed_count = retriever.index_chunks(all_chunks)

    print(f"Successfully indexed {indexed_count} chunks.")

    # Step 4: Initialize the LLM
    llm = OllamaClient()

    # Step 5: Initialize the Research Agent
    agent = ResearchAgent(
        retriever=retriever,
        llm=llm
    )

    # Step 6: Initialize the Research Workflow
    workflow = ResearchWorkflow(
        agent=agent,
        llm=llm
    )

    # Step 7: Receive the user's request
    query = query = (
    "How does the Minimum Model Uncertainty "
    "criterion evaluate Bayesian Network structure quality "
    "using misclassification error?"
    )

    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)
    print(query)

    # Step 8: Execute the research workflow
    workflow_result = workflow.run(query)

    # Step 9: Display the workflow result
    print("\n" + "=" * 70)
    print("WORKFLOW RESULT")
    print("=" * 70)

    print(f"Selected Tool: {workflow_result['tool']}")

    if workflow_result["tool"] == "list_documents":

        documents = workflow_result["documents"]

        print(f"\nAvailable Academic Papers: {len(documents)}")

        for rank, document in enumerate(documents, start=1):

            print(f"\nDocument {rank}")
            print("-" * 70)
            print(f"Name: {document['source']}")
            print(f"Pages: {document['pages']}")
            print(f"Chunks: {document['chunks']}")

    elif workflow_result["tool"] == "search_documents":

        print("\nRESEARCH ANSWER")
        print("-" * 70)

        print(workflow_result["answer"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
