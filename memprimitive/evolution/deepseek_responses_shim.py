"""Tiny Responses API shim for Codex -> DeepSeek chat-completions runs."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


TEXT_PART_TYPES = {"input_text", "output_text", "text"}


def _normalize_responses_input_items(payload: dict[str, Any]) -> list[Any]:
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": raw_input}],
            }
        ]
    if isinstance(raw_input, list):
        return raw_input
    return list(raw_input or [])


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_debug_payload(name: str, payload: dict[str, Any]) -> None:
    log_dir = os.environ.get("DEEPSEEK_SHIM_LOG_DIR", "").strip()
    if not log_dir:
        return
    path = os.path.join(log_dir, f"{time.time_ns()}_{name}.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES:
            parts.append(str(part.get("text", "")))
    return "\n".join(part for part in parts if part)


def _responses_input_to_chat_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = str(payload.get("instructions") or "").strip()
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_items = _normalize_responses_input_items(payload)
    index = 0
    while index < len(input_items):
        item = input_items[index]
        if not isinstance(item, dict):
            index += 1
            continue
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _text_from_content(item.get("content"))})
            index += 1
        elif item_type == "function_call":
            tool_calls: list[dict[str, Any]] = []
            while index < len(input_items):
                call_item = input_items[index]
                if not isinstance(call_item, dict) or call_item.get("type") != "function_call":
                    break
                call_id = str(call_item.get("call_id") or call_item.get("id") or f"call_{uuid.uuid4().hex}")
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(call_item.get("name") or ""),
                            "arguments": str(call_item.get("arguments") or "{}"),
                        },
                    }
                )
                index += 1
            assistant_text: list[str] = []
            while index < len(input_items):
                text_item = input_items[index]
                if not (
                    isinstance(text_item, dict)
                    and text_item.get("type") == "message"
                    and text_item.get("role") == "assistant"
                ):
                    break
                text = _text_from_content(text_item.get("content"))
                if text:
                    assistant_text.append(text)
                index += 1
            messages.append({"role": "assistant", "content": "\n".join(assistant_text) or None, "tool_calls": tool_calls})
        elif item_type == "function_call_output":
            call_outputs: list[dict[str, Any]] = []
            while index < len(input_items):
                output_item = input_items[index]
                if not isinstance(output_item, dict) or output_item.get("type") != "function_call_output":
                    break
                output = output_item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                call_outputs.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(output_item.get("call_id") or ""),
                        "content": output,
                    }
                )
                index += 1
            messages.extend(call_outputs)
        else:
            index += 1
    return messages


def _responses_tools_to_chat_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _tool_choice_to_chat(value: Any) -> Any:
    if value in (None, "auto", "none", "required"):
        return value
    if isinstance(value, dict):
        name = value.get("name") or value.get("function", {}).get("name")
        if name:
            return {"type": "function", "function": {"name": str(name)}}
    return "auto"


def _build_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    chat_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": _responses_input_to_chat_messages(payload),
        "stream": False,
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    tools = _responses_tools_to_chat_tools(payload)
    if tools:
        chat_payload["tools"] = tools
        chat_payload["tool_choice"] = _tool_choice_to_chat(payload.get("tool_choice", "auto"))
    if payload.get("temperature") is not None:
        chat_payload["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        chat_payload["top_p"] = payload["top_p"]
    if payload.get("max_output_tokens") is not None:
        chat_payload["max_tokens"] = payload["max_output_tokens"]
    return chat_payload


def _call_deepseek(payload: dict[str, Any], *, api_key: str, base_url: str, timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=_json_bytes(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _message_to_response_output(message: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        outputs.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
            }
        )
    content = message.get("content")
    if content:
        outputs.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": str(content), "annotations": []}],
            }
        )
    return outputs


def _chat_to_response(chat: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = _message_to_response_output(message)
    usage = chat.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
        },
    }


def _sse_events(response: dict[str, Any]) -> bytes:
    events: list[dict[str, Any]] = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
    for index, item in enumerate(response.get("output") or []):
        if item.get("type") == "message":
            text = str((item.get("content") or [{}])[0].get("text") or "")
            item_id = str(item.get("id"))
            events.extend(
                [
                    {"type": "response.output_item.added", "output_index": index, "item": {**item, "status": "in_progress", "content": []}},
                    {
                        "type": "response.content_part.added",
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                    {"type": "response.output_text.delta", "item_id": item_id, "output_index": index, "content_index": 0, "delta": text},
                    {"type": "response.output_text.done", "item_id": item_id, "output_index": index, "content_index": 0, "text": text},
                    {
                        "type": "response.content_part.done",
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    },
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ]
            )
        else:
            events.extend(
                [
                    {"type": "response.output_item.added", "output_index": index, "item": item},
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ]
            )
    events.append({"type": "response.completed", "response": response})
    return ("".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events) + "data: [DONE]\n\n").encode("utf-8")


class ShimHandler(BaseHTTPRequestHandler):
    upstream_base_url = "https://api.deepseek.com/v1"
    upstream_timeout = 120.0

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        request_path = self.path.split("?", 1)[0].rstrip("/")
        if request_path.endswith("/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-v4-pro",
                            "object": "model",
                            "owned_by": "deepseek",
                        },
                        {
                            "id": "deepseek-v4-flash",
                            "object": "model",
                            "owned_by": "deepseek",
                        },
                    ],
                },
            )
            return
        if request_path in {"", "/v1", "/health"}:
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/responses"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("content-length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            _write_debug_payload("responses_request", payload)
            api_key = os.environ["DEEPSEEK_API_KEY"]
            chat_payload = _build_chat_payload(payload)
            _write_debug_payload("chat_request", chat_payload)
            chat = _call_deepseek(
                chat_payload,
                api_key=api_key,
                base_url=self.upstream_base_url,
                timeout=self.upstream_timeout,
            )
            _write_debug_payload("chat_response", chat)
            response = _chat_to_response(chat, str(payload.get("model") or "deepseek"))
            _write_debug_payload("responses_response", response)
        except KeyError:
            self._send_json(500, {"error": {"message": "DEEPSEEK_API_KEY is not set"}})
            return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self._send_json(exc.code, {"error": {"message": body or str(exc)}})
            return
        except Exception as exc:  # pragma: no cover - keeps the shim debuggable during live runs.
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        if payload.get("stream", False):
            body = _sse_events(response)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(200, response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upstream-base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    ShimHandler.upstream_base_url = args.upstream_base_url
    ShimHandler.upstream_timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), ShimHandler)
    print(f"deepseek responses shim listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
