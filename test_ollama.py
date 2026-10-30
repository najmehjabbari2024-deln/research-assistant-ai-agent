import requests


OLLAMA_URL = "http://localhost:11434/api/chat"


def test_ollama_connection():

    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {
                "role": "user",
                "content": "Explain what a Bayesian Network is in simple academic English."
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    answer = result["message"]["content"]

    print("\nQwen Response:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    test_ollama_connection()