
import httpx

from app.core.config import get_settings


class OllamaClient:
    def __init__(self):
        settings = get_settings()

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            timeout=120.0,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]

    
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Send a chat request to Ollama.
        Optionally provide tools for function calling.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if tools is not None:
            payload["tools"] = tools

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=180.0,
        )

        response.raise_for_status()

        data = response.json()
        message = data.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "Invalid response received from Ollama."
            )

        return message

_ollama_client = None


def get_ollama_client() -> OllamaClient:
    global _ollama_client

    if _ollama_client is None:
        _ollama_client = OllamaClient()

    return _ollama_client
