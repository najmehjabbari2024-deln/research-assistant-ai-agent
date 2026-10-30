from src.workflow.research_workflow import ResearchWorkflow


class MockAgent:
    """
    Simulates the Research Agent.
    """

    def run(self, user_query):
        return {
            "tool": "search_documents",
            "tool_result": [
                {
                    "source": "sample_paper.pdf",
                    "page": 3,
                    "chunk_id": 1,
                    "text": "Bayesian Networks represent probabilistic relationships between variables.",
                    "score": 0.85
                }
            ]
        }


class MockLLM:
    """
    Simulates the language model and records its input.
    """

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        self.received_prompt = prompt

        return "Bayesian Networks represent probabilistic relationships between variables. [E1]"


def test_research_workflow_generates_answer():

    mock_agent = MockAgent()
    mock_llm = MockLLM()

    workflow = ResearchWorkflow(
        agent=mock_agent,
        llm=mock_llm
    )

    query = "What are Bayesian Networks?"

    result = workflow.run(query)

    assert result["tool"] == "search_documents"

    assert result["request"] == query

    assert result["answer"] is not None

    assert "[EVIDENCE_ID: E1]" in mock_llm.received_prompt

    assert "sample_paper.pdf" in mock_llm.received_prompt

    assert "Bayesian Networks" in mock_llm.received_prompt
def test_research_workflow_lists_documents():

    class DocumentListAgent:

        def run(self, user_query):
            return {
                "tool": "list_documents",
                "tool_result": [
                    {
                        "source": "paper_1.pdf",
                        "pages": 22,
                        "chunks": 124
                    },
                    {
                        "source": "paper_2.pdf",
                        "pages": 40,
                        "chunks": 130
                    }
                ]
            }

    mock_llm = MockLLM()

    workflow = ResearchWorkflow(
        agent=DocumentListAgent(),
        llm=mock_llm
    )

    query = "Which academic papers are available?"

    result = workflow.run(query)

    assert result["tool"] == "list_documents"

    assert result["request"] == query

    assert len(result["documents"]) == 2

    assert result["documents"][0]["source"] == "paper_1.pdf"

    assert mock_llm.received_prompt is None