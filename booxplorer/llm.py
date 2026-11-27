"""Client for the existing local OpenAI-compatible chat runtime."""

from dataclasses import dataclass
import json
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LocalChat:
    endpoint: str
    model: str
    timeout: float = 240
    disable_thinking: bool = False

    def __call__(self, instructions: str, data: dict) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2700,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
            ],
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            choice = json.load(response)["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError(f"incomplete chat response: {choice['finish_reason']}")
        output = json.loads(choice["message"]["content"])
        if not isinstance(output, dict):
            raise ValueError("chat response must be a JSON object")
        return output
