
import streamlit as st
import tempfile
import re
from pathlib import Path

from src.document_processing.pdf_processor import extract_text_from_pdf
from src.document_processing.text_chunker import chunk_pages
from src.retrieval.retriever import TfidfRetriever
from src.llm.ollama_client import OllamaClient
from src.agents.research_agent import ResearchAgent
from src.workflow.research_workflow import ResearchWorkflow


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# 2. Citation Helper
# --------------------------------------------------

def make_citations_clickable(answer):
    """
    Convert evidence citations such as [E1] into clickable links.
    """

    pattern = r"\[(E\d+)\]"

    def replace_citation(match):

        evidence_id = match.group(1)

        return (
            f'<a href="#evidence-{evidence_id}" '
            f'style="color:#7C6CF0; font-weight:bold; '
            f'text-decoration:none;">'
            f'[{evidence_id}]</a>'
        )

    return re.sub(pattern, replace_citation, answer)


# --------------------------------------------------
# 3. Application State
# --------------------------------------------------

if "all_pages" not in st.session_state:
    st.session_state.all_pages = []

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = OllamaClient()

if "agent" not in st.session_state:
    st.session_state.agent = None

if "workflow" not in st.session_state:
    st.session_state.workflow = None

# Store the latest research result
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Store the question associated with the latest result
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# --------------------------------------------------
# 4. Application Header
# --------------------------------------------------

st.title("📚 Research Assistant AI")

st.caption(
    "An AI-powered assistant for scientific literature analysis."
)

st.divider()


# --------------------------------------------------
# 5. Sidebar - Document Library
# --------------------------------------------------

with st.sidebar:

    st.header("Document Library")

    uploaded_files = st.file_uploader(
        "Upload academic PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --------------------------------------------------
    # Process Uploaded PDF Documents
    # --------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_name = uploaded_file.name

            # Skip previously processed files
            if file_name in st.session_state.processed_files:
                continue

            try:

                with st.spinner(f"Processing {file_name}..."):

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())

                        temp_pdf_path = Path(temp_file.name)

                    try:

                        # Extract text from PDF
                        pages = extract_text_from_pdf(
                            temp_pdf_path,
                            source_name=file_name
                        )

                        if not pages:
                            raise ValueError(
                                "No extractable text found in this PDF."
                            )

                        # Split extracted text into chunks
                        chunks = chunk_pages(
                            pages,
                            chunk_size=1000,
                            overlap=200
                        )

                        if not chunks:
                            raise ValueError(
                                "No text chunks could be generated."
                            )

                        # Store processed data
                        st.session_state.all_pages.extend(pages)

                        st.session_state.all_chunks.extend(chunks)

                        st.session_state.processed_files.add(file_name)

                    finally:

                        # Remove temporary PDF file
                        temp_pdf_path.unlink(missing_ok=True)

                st.success(
                    f"Successfully processed: {file_name}"
                )

            except Exception as error:

                st.error(
                    f"Error processing {file_name}: {error}"
                )

    st.divider()

    # --------------------------------------------------
    # Build or Update the Retriever
    # --------------------------------------------------

    if st.session_state.all_chunks:

        with st.spinner("Indexing research documents..."):

            retriever = TfidfRetriever()

            retriever.index_chunks(
                st.session_state.all_chunks
            )

            st.session_state.retriever = retriever

            # Initialize Agent
            agent = ResearchAgent(
                retriever=st.session_state.retriever,
                llm=st.session_state.llm
            )

            st.session_state.agent = agent

            # Initialize Workflow
            workflow = ResearchWorkflow(
                agent=st.session_state.agent,
                llm=st.session_state.llm
            )

            st.session_state.workflow = workflow

    # --------------------------------------------------
    # Library Statistics
    # --------------------------------------------------

    st.subheader("Library Statistics")

    st.metric(
        "Documents Indexed",
        len(st.session_state.processed_files)
    )

    st.metric(
        "Total Pages",
        len(st.session_state.all_pages)
    )

    st.metric(
        "Total Chunks",
        len(st.session_state.all_chunks)
    )


# --------------------------------------------------
# 6. Main Area - Research Query
# --------------------------------------------------

st.header("Research Workspace")

st.write(
    "Ask a scientific question based on your uploaded papers."
)

user_query = st.text_area(
    "Research Question",
    placeholder="e.g. How does Bayesian Network structure learning work?",
    height=120
)

search_button = st.button(
    "🔍 Search & Generate Answer",
    use_container_width=True
)


# --------------------------------------------------
# 7. Results Section
# --------------------------------------------------

st.divider()

st.header("Research Results")


# --------------------------------------------------
# Execute Research Workflow
# --------------------------------------------------

if search_button:

    if not user_query.strip():

        st.warning("Please enter a research question.")

    elif st.session_state.workflow is None:

        st.warning(
            "Please upload at least one academic PDF document first."
        )

    else:

        try:

            with st.spinner(
                "Analyzing documents and generating your answer..."
            ):

                result = st.session_state.workflow.run(
                    user_query
                )

            # Save the latest result
            st.session_state.last_result = result

            # Save the corresponding question
            st.session_state.last_query = user_query

        except Exception as error:

            st.error(
                f"An error occurred during research: {error}"
            )


# --------------------------------------------------
# Display Saved Results
# --------------------------------------------------

if st.session_state.last_result is not None:

    result = st.session_state.last_result

    # Display the original research question
    st.caption(
        f"Research Question: {st.session_state.last_query}"
    )

    # --------------------------------------------------
    # Search Results
    # --------------------------------------------------

    if result["tool"] == "search_documents":

        st.subheader("Research Answer")

        # Convert evidence citations into clickable links
        clickable_answer = make_citations_clickable(
            result["answer"]
        )

        st.markdown(
            clickable_answer,
            unsafe_allow_html=True
        )

        st.divider()

        # --------------------------------------------------
        # Supporting Evidence
        # --------------------------------------------------

        st.subheader("Supporting Evidence")

        for index, evidence in enumerate(
            result["evidence"],
            start=1
        ):

            evidence_id = f"E{index}"

            # Create an anchor for each evidence item
            st.markdown(
                f'<div id="evidence-{evidence_id}"></div>',
                unsafe_allow_html=True
            )

            with st.expander(
                f"Evidence {evidence_id} | "
                f"{evidence['source']} | "
                f"Page {evidence['page']}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.caption("SOURCE DOCUMENT")

                    st.write(evidence["source"])

                with col2:

                    st.caption("RETRIEVAL RELEVANCE SCORE")

                    st.metric(
                        label="Similarity",
                        value=f"{evidence['score']:.4f}"
                    )

                st.divider()

                st.markdown("**Retrieved Text**")

                st.write(evidence["text"])

    # --------------------------------------------------
    # Document Library Results
    # --------------------------------------------------

    elif result["tool"] == "list_documents":

        st.subheader("Available Research Documents")

        for document in result["documents"]:

            st.write(
                f"**Document:** {document['source']}"
            )

            st.write(
                f"Pages: {document['pages']} | "
                f"Chunks: {document['chunks']}"
            )

            st.divider()
