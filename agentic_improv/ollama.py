"""Ollama REST adapter restricted to the local daemon."""

import json
import re
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from time import perf_counter


OLLAMA_URL = "http://localhost:11434"
THINKING_BLOCK = re.compile(r"<\s*think\b[^>]*>.*?(?:<\s*/\s*think\s*>|\Z)", re.I | re.S)
THINKING_END = re.compile(r"<\s*/\s*think\s*>", re.I)
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
    "additionalProperties": False,
}
REPLY_INSTRUCTION = (
    "Return a JSON object with one field, reply. In reply, write only the final message "
    "addressed to the other agent. Put any formatting required by the experiment inside that string."
)


class OllamaError(RuntimeError):
    """An error suitable for display in the dashboard."""


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None


def visible_text(content: str) -> str:
    content = THINKING_BLOCK.sub("", content)
    # Some templates return a closing tag even when the opening tag was prefilled.
    return THINKING_END.split(content)[-1].strip()


def token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


class OllamaClient:
    def _request(self, path: str, payload: dict | None = None, *, timeout: float) -> dict:
        connection = HTTPConnection("localhost", 11434, timeout=timeout)
        try:
            if payload is None:
                connection.request("GET", path)
            else:
                connection.request(
                    "POST", path, json.dumps(payload).encode("utf-8"),
                    {"Content-Type": "application/json"},
                )
            with connection.getresponse() as response:
                if response.status == 404:
                    raise OllamaError("Ollama could not find the requested model or API endpoint. Refresh models and check your Ollama version.")
                if response.status >= 500:
                    raise OllamaError(f"Ollama returned HTTP {response.status}. Check the Ollama server log and available memory.")
                if response.status != 200:
                    raise OllamaError(f"Ollama returned HTTP {response.status}. Check that the selected model supports chat.")
                data = json.loads(response.read())
        except TimeoutError as exc:
            raise OllamaError("The local Ollama request timed out. Check that Ollama is responsive and try again.") from exc
        except (OSError, HTTPException) as exc:
            raise OllamaError(f"Could not reach Ollama at {OLLAMA_URL}. Start Ollama and refresh the model list.") from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc
        finally:
            connection.close()
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an unexpected response.")
        return data

    def _installed_models(self) -> dict[str, dict]:
        entries = self._request("/api/tags", timeout=3).get("models")
        if not isinstance(entries, list):
            raise OllamaError("Ollama returned an invalid model list.")
        models = {}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("remote_model") or entry.get("remote_host"):
                continue
            name = entry.get("name") or entry.get("model")
            if isinstance(name, str) and name and not name.lower().endswith((":cloud", "-cloud")):
                models[name] = entry
        return models

    def list_models(self) -> list[str]:
        return sorted(self._installed_models(), key=str.casefold)

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> Reply:
        entry = self._installed_models().get(model)
        if entry is None:
            raise OllamaError(f"The selected model ({model}) is unavailable locally. Refresh the model list.")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        details = entry.get("details") or {}
        model_family = model.rsplit("/", 1)[-1].partition(":")[0].lower()
        qwen3 = model_family == "qwen3" or details.get("family") == "qwen3"
        if qwen3:
            # Some installed Qwen3 templates start an unmarked thinking block even with think=False.
            # Constrain decoding to a reply field so that free-form output cannot enter the transcript.
            payload["format"] = REPLY_SCHEMA
            payload["messages"] = [{
                "role": "system",
                "content": f"{messages[0]['content']}\n\n{REPLY_INSTRUCTION}",
            }, *messages[1:]]
        started = perf_counter()
        data = self._request("/api/chat", payload, timeout=300)
        elapsed_ms = round((perf_counter() - started) * 1000, 1)
        message = data.get("message")
        if data.get("done") is not True or not isinstance(message, dict):
            raise OllamaError("Ollama did not return a complete chat response.")
        content = message.get("content")
        if qwen3 and isinstance(content, str):
            try:
                envelope = json.loads(content)
                content = envelope.get("reply") if isinstance(envelope, dict) and set(envelope) == {"reply"} else None
            except ValueError:
                content = None
        text = visible_text(content) if isinstance(content, str) else ""
        if not text:
            hint = " Increase Max output tokens and try again." if data.get("done_reason") == "length" else " Try a different chat model or prompt."
            raise OllamaError(f"{model} returned no visible reply.{hint}")
        reported_model = data.get("model")
        return Reply(
            text, reported_model if isinstance(reported_model, str) and reported_model else model,
            elapsed_ms, token_count(data.get("prompt_eval_count")), token_count(data.get("eval_count")),
        )
