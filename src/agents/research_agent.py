
import json


class ResearchAgent:
    """
    A research assistant agent that selects and
    executes tools based on user requests.
    """

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

        self.tools = {
            "search_documents": self.search_documents,
            "list_documents": self.list_documents
        }

    def search_documents(self, query, top_k=5):
        """
        Search academic documents for relevant evidence.
        """

        return self.retriever.search(
            query=query,
            top_k=top_k
        )

    def list_documents(self):
        """
        Return the list of indexed documents and their statistics.
        """

        documents = {}

        for chunk in self.retriever.chunks:

            source = chunk["source"]

            if source not in documents:
                documents[source] = {
                    "pages": set(),
                    "chunks": 0
                }

            documents[source]["pages"].add(chunk["page"])
            documents[source]["chunks"] += 1

        results = []

        for source, info in documents.items():

            results.append({
                "source": source,
                "pages": len(info["pages"]),
                "chunks": info["chunks"]
            })

        return results

    def select_tool(self, user_query):
        """
        Ask the LLM to select the appropriate tool.
        """

        tool_selection_prompt = f"""
You are a tool selection assistant.

Available tools:

1. search_documents:
   Searches academic PDF documents for evidence.
   Use this tool when the user asks a research question.

2. list_documents:
   Returns the names and statistics of indexed documents.
   Use this tool when the user asks which documents
   or papers are available in the system.

User request:
{user_query}

Return valid JSON only.

For search_documents, use this format:
{{
    "tool": "search_documents",
    "tool_input": {{
        "query": "the search query"
    }}
}}

For list_documents, use this format:
{{
    "tool": "list_documents",
    "tool_input": {{}}
}}
"""

        response = self.llm.generate(
            prompt=tool_selection_prompt,
            system_prompt=(
                "You are a tool selection component. "
                "Return valid JSON only. Do not add explanations."
            ),
            temperature=0
        )

        try:
            decision = json.loads(response)

        except json.JSONDecodeError as error:
            raise ValueError(
                "The LLM returned invalid JSON."
            ) from error

        tool_name = decision.get("tool")

        if tool_name not in self.tools:
            raise ValueError(
                f"Unknown tool selected: {tool_name}"
            )

        return decision

    def run(self, user_query):
        """
        Select and execute a tool based on the user request.
        """

        decision = self.select_tool(user_query)

        tool_name = decision["tool"]
        tool_input = decision["tool_input"]

        tool = self.tools[tool_name]

        tool_result = tool(**tool_input)

        return {
            "tool": tool_name,
            "tool_result": tool_result
        }
