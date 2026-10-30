
from src.retrieval.context_builder import build_context
from src.llm.prompts import RESEARCH_ASSISTANT_SYSTEM_PROMPT


class ResearchWorkflow:
    """
    Orchestrates the execution of the research assistant pipeline.

    Responsibilities:
    1. Receive the user's request.
    2. Delegate tool selection and execution to the agent.
    3. Process the tool output.
    4. Generate a research answer when evidence is retrieved.
    """

    def __init__(self, agent, llm):

        self.agent = agent
        self.llm = llm

    def run(self, user_query):
        """
        Execute the complete research workflow.
        """

        # Step 1: Let the agent select and execute a tool
        agent_result = self.agent.run(user_query)

        tool_name = agent_result["tool"]
        tool_result = agent_result["tool_result"]

        # Step 2: Handle the output based on the selected tool
        if tool_name == "search_documents":

            # Build structured evidence context
            context = build_context(
                tool_result,
                max_context_chars=6000
            )

            # Prepare the research prompt
            prompt = f"""
Research Question:
{user_query}

Retrieved Scientific Evidence:
{context}

Answer the research question using the provided evidence.
Include the relevant evidence IDs in your answer.
"""

            # Generate the research answer
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=RESEARCH_ASSISTANT_SYSTEM_PROMPT,
                temperature=0.2
            )

            return {
                "request": user_query,
                "tool": tool_name,
                "evidence": tool_result,
                "answer": answer
            }

        elif tool_name == "list_documents":

            return {
                "request": user_query,
                "tool": tool_name,
                "documents": tool_result
            }

        else:

            raise ValueError(
                f"Unsupported tool: {tool_name}"
            )
