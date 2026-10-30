from src.llm.ollama_client import OllamaClient


def test_generate():

    client = OllamaClient()

    answer = client.generate(
        prompt="Explain what a Bayesian Network is in simple academic English.",
        system_prompt=(
            "You are an academic research assistant. "
            "Provide clear, accurate, and concise explanations."
        )
    )

    print("\nGenerated Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    test_generate()