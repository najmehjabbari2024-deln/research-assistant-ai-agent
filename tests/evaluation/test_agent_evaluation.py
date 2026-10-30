
from src.agents.research_agent import ResearchAgent


class MockLLM:
    """
    Simulates LLM responses for testing.
    """

    def __init__(self, response):
        self.response = response

    def generate(self, *args, **kwargs):
        return self.response


class MockRetriever:
    """
    Simulates a document retriever.
    """

    def search(self, query, top_k=5):
        return [
            {
                "source": "sample_paper.pdf",
                "page": 1,
                "chunk_id": 1,
                "text": "Sample research evidence.",
                "score": 0.85
            }
        ]


def test_search_documents_tool_selection():

    mock_llm = MockLLM(
        """
        {
            "tool": "search_documents",
            "tool_input": {
                "query": "Bayesian Network structure learning"
            }
        }
        """
    )

    agent = ResearchAgent(
        retriever=MockRetriever(),
        llm=mock_llm
    )

    decision = agent.select_tool(
        "Find evidence about Bayesian Network structure learning."
    )

    assert decision["tool"] == "search_documents"


def test_list_documents_tool_selection():

    mock_llm = MockLLM(
        """
        {
            "tool": "list_documents",
            "tool_input": {}
        }
        """
    )

    agent = ResearchAgent(
        retriever=MockRetriever(),
        llm=mock_llm
    )

    decision = agent.select_tool(
        "Which academic papers are available?"
    )

    assert decision["tool"] == "list_documents"
def test_agent_executes_selected_tool():

    mock_llm = MockLLM(
        """
        {
            "tool": "search_documents",
            "tool_input": {
                "query": "Bayesian Network structure learning"
            }
        }
        """
    )

    mock_retriever = MockRetriever()

    agent = ResearchAgent(
        retriever=mock_retriever,
        llm=mock_llm
    )

    result = agent.run(
        "Find evidence about Bayesian Network structure learning."
    )

    assert result["tool"] == "search_documents"

    assert len(result["tool_result"]) == 1

    assert result["tool_result"][0]["source"] == "sample_paper.pdf"

import pytest


def test_agent_handles_invalid_json():

    mock_llm = MockLLM(
        "This is not a valid JSON response."
    )

    agent = ResearchAgent(
        retriever=MockRetriever(),
        llm=mock_llm
    )

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        agent.select_tool(
            "Find evidence about Bayesian Network structure learning."
        )

def test_agent_handles_unknown_tool():

    mock_llm = MockLLM(
        """
        {
            "tool": "delete_documents",
            "tool_input": {}
        }
        """
    )

    agent = ResearchAgent(
        retriever=MockRetriever(),
        llm=mock_llm
    )

    with pytest.raises(ValueError, match="Unknown tool selected"):
        agent.select_tool(
            "Delete all documents."
        )