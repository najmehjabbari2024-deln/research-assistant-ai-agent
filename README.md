# Research Assistant AI Agent
## Project Overview

Research Assistant AI Agent is a local AI-powered application designed to help researchers analyze scientific literature more efficiently.

Users can upload academic PDF documents, ask research-related questions, and receive AI-generated answers grounded in the content of their uploaded papers.

The system retrieves relevant text passages from research documents and uses a Large Language Model (LLM) to generate structured answers with references to the original documents and page numbers.

The application also provides a document management interface where users can upload, index, and explore their research papers.

## Key Features

- Upload and process multiple academic PDF documents.
- Extract text while preserving document and page metadata.
- Split documents into overlapping text chunks.
- Retrieve relevant passages using TF-IDF and cosine similarity.
- Generate evidence-based answers using a locally running LLM.
- Route user requests to appropriate tools through an LLM-assisted agent.
- Display supporting evidence with source document and page information.
- Explore indexed documents through an interactive Streamlit interface.

## System Architecture

The application follows a modular architecture that separates document processing, information retrieval, language model interaction, agent-based tool routing, and workflow orchestration.

### Main Components

| Component | Responsibility |
|---|---|
| Document Processing | Extracts text and page metadata from academic PDF files. |
| Text Chunking | Splits extracted text into manageable, overlapping chunks. |
| Retrieval System | Indexes text chunks using TF-IDF and retrieves relevant passages using cosine similarity. |
| Context Builder | Organizes retrieved passages into structured evidence with source and page metadata. |
| LLM Client | Communicates with the local Qwen model through the Ollama API. |
| Research Agent | Selects and executes the appropriate tool based on the user's request. |
| Workflow Orchestrator | Coordinates tool execution, evidence retrieval, and answer generation. |
| Streamlit Interface | Provides the user interface for document uploads and research queries. |

### Processing Pipeline

1. **Document Ingestion:** Users upload academic PDF files.
2. **Text Extraction:** PyMuPDF extracts document text while preserving page information.
3. **Chunking:** Extracted text is divided into overlapping chunks.
4. **Indexing:** TF-IDF converts text chunks into numerical vectors.
5. **Tool Selection:** The agent uses the LLM to determine which available tool should handle the request.
6. **Information Retrieval:** The selected search tool retrieves the most relevant chunks.
7. **Context Construction:** Retrieved passages are formatted as structured evidence.
8. **Answer Generation:** The LLM generates an answer using the retrieved evidence.
9. **Presentation:** The interface displays the answer and its supporting evidence.

## Technology Stack

The project is developed using the following technologies:

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| PyMuPDF | PDF text extraction and page-level metadata processing |
| Scikit-learn | TF-IDF vectorization and cosine similarity-based retrieval |
| NumPy | Numerical computing support |
| Pandas | Data manipulation and analysis |
| Ollama | Local LLM serving and model inference |
| Qwen 2.5 (1.5B) | Local language model for tool selection and answer generation |
| Requests | HTTP communication with the Ollama API |
| Streamlit | Interactive web application interface |
| Pytest | Automated testing and evaluation |

### Core AI Components

- **Retrieval:** TF-IDF with unigram and bigram features.
- **Similarity Measurement:** Cosine similarity.
- **Language Model:** Qwen 2.5 (1.5B).
- **LLM Runtime:** Ollama.
- **Agent Routing:** LLM-assisted JSON-based tool selection.
- **Application Framework:** Streamlit.

## Installation & Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.11
- Git
- Ollama

### 1. Clone the Repository
git clone https://github.com/najmehjabbari2024-deln/research-assistant-ai-agent.git
cd research-assistant-ai-agent
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Research Assistant AI Agent"
