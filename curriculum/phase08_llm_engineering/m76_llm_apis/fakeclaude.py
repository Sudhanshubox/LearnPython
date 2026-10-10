"""fakeclaude: a fake Claude API for offline tests, given for every Phase 8 module.

It plugs a scripted HTTP transport into the REAL `anthropic` SDK, so your code runs exactly
as it would against the real API, but replies come from a script and nothing leaves your
computer. Every request your code sends is recorded so tests can check it.

    fake = FakeClaude()
    fake.add_text("Hello!")                          # the next reply
    client = fake.client()                           # a real anthropic.Anthropic
    client.messages.create(model=..., max_tokens=..., messages=[...])
    fake.requests[-1]["body"]                        # what was sent

Replies are used in order. Instead of a script you can set `fake.responder`, a function
from the request body (a dict) to a reply made with text_reply / tool_reply / error_reply.
Streaming requests (`client.messages.stream(...)`) get the same reply as server-sent events.
"""

import itertools
import json

import anthropic
import httpx2

MODEL = "claude-opus-5-5"


def usage(input_tokens=10, output_tokens=5, cache_read=0, cache_write=0):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write}


def text_reply(text, stop_reason="end_turn", input_tokens=10, output_tokens=5, cache_read=0, cache_write=0):
    """A reply containing one text block."""
    return {"kind": "message", "content": [{"type": "text", "text": text}], "stop_reason": stop_reason,
            "usage": usage(input_tokens, output_tokens, cache_read, cache_write)}


def tool_reply(calls, text=None, input_tokens=10, output_tokens=5):
    """A reply asking for tool calls. calls: a list of (name, input_dict) or (id, name, input_dict)."""
    content = [{"type": "text", "text": text}] if text else []
    for call in calls:
        if len(call) == 2:
            call = (None, *call)
        tool_id, name, tool_input = call
        content.append({"type": "tool_use", "id": tool_id, "name": name, "input": tool_input})
    return {"kind": "message", "content": content, "stop_reason": "tool_use",
            "usage": usage(input_tokens, output_tokens)}


def refusal_reply(category="cyber"):
    """A reply where the model declined (stop_reason "refusal")."""
    reply = text_reply("", stop_reason="refusal")
    reply["content"] = []
    reply["stop_details"] = {"type": "refusal", "category": category, "explanation": "Declined."}
    return reply


ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 429: "rate_limit_error", 500: "api_error", 529: "overloaded_error"}


def error_reply(status, message="error", retry_after=None):
    """An HTTP error response, e.g. error_reply(429) for a rate limit."""
    return {"kind": "error", "status": status, "message": message, "retry_after": retry_after}


class FakeClaude:
    def __init__(self, responder=None):
        self.queue = []
        self.requests = []
        self.responder = responder
        self._ids = itertools.count(1)

    # --- scripting -------------------------------------------------------------------
    def add(self, *replies):
        self.queue.extend(replies)
        return self

    def add_text(self, text, **kwargs):
        return self.add(text_reply(text, **kwargs))

    def add_tool_use(self, calls, text=None, **kwargs):
        return self.add(tool_reply(calls, text, **kwargs))

    def add_error(self, status, message="error", retry_after=None):
        return self.add(error_reply(status, message, retry_after))

    # --- the client ------------------------------------------------------------------
    def client(self, max_retries=0, **kwargs):
        transport = httpx2.MockTransport(self._handle)
        return anthropic.Anthropic(api_key="sk-ant-test", max_retries=max_retries,
                                   http_client=anthropic.DefaultHttpxClient(transport=transport), **kwargs)

    @property
    def bodies(self):
        return [r["body"] for r in self.requests if r["path"].endswith("/v1/messages")]

    def _next(self, body):
        if self.queue:
            return self.queue.pop(0)
        if self.responder is not None:
            return self.responder(body)
        return error_reply(500, "FakeClaude: no reply was scripted for this request")

    def _handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": max(1, len(json.dumps(body)) // 4)})
        reply = self._next(body)
        if reply["kind"] == "error":
            headers = {"retry-after": str(reply["retry_after"])} if reply["retry_after"] is not None else {}
            error = {"type": "error", "error": {"type": ERROR_TYPES.get(reply["status"], "api_error"),
                                               "message": reply["message"]}}
            return httpx2.Response(reply["status"], json=error, headers=headers)
        message = self._message(reply, body)
        if body.get("stream"):
            return httpx2.Response(200, text=self._sse(message), headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=message)

    def _message(self, reply, body):
        content = []
        for block in reply["content"]:
            block = dict(block)
            if block["type"] == "tool_use" and block["id"] is None:
                block["id"] = f"toolu_{next(self._ids):04d}"
            content.append(block)
        message = {"id": f"msg_{next(self._ids):04d}", "type": "message", "role": "assistant",
                   "model": body.get("model", MODEL), "content": content, "stop_reason": reply["stop_reason"],
                   "stop_sequence": None, "usage": reply["usage"]}
        if "stop_details" in reply:
            message["stop_details"] = reply["stop_details"]
        return message

    @staticmethod
    def _sse(message):
        start = {**message, "content": [], "stop_reason": None,
                 "usage": {**message["usage"], "output_tokens": 0}}
        events = [("message_start", {"type": "message_start", "message": start})]
        for i, block in enumerate(message["content"]):
            if block["type"] == "text":
                events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {"type": "text", "text": ""}}))
                text = block["text"]
                for j in range(0, len(text), 4):
                    events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                           "delta": {"type": "text_delta", "text": text[j:j + 4]}}))
            else:
                events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                       "content_block": {**block, "input": {}}}))
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                       "delta": {"type": "input_json_delta",
                                                                 "partial_json": json.dumps(block["input"])}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        delta = {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                 "usage": {"output_tokens": message["usage"]["output_tokens"]}}
        events += [("message_delta", delta), ("message_stop", {"type": "message_stop"})]
        return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
