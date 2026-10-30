
import requests


class OllamaClient:
    """
    A client for communicating with a local Ollama LLM.
    """

    def __init__(
        self,
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434"
    ):

        self.model = model
        self.api_url = f"{base_url}/api/chat"

    def generate(
        self,
        prompt,
        system_prompt=None,
        temperature=0.2
    ):

        # Build conversation messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Prepare API request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        # Send request to Ollama
        try:

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=180
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError as error:

            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Please make sure Ollama is running."
            ) from error

        except requests.exceptions.Timeout as error:

            raise RuntimeError(
                "The LLM request timed out. "
                "The model may need more time to respond."
            ) from error

        except requests.exceptions.HTTPError as error:

            raise RuntimeError(
                f"Ollama returned an HTTP error: {error}"
            ) from error

        # Parse model response
        result = response.json()

        return result["message"]["content"]
