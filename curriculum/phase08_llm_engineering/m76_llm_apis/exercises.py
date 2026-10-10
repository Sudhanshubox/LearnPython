"""m76 exercises: calling the Claude API well.

Every function receives a `client` (an anthropic.Anthropic). The tests pass a fake one from
fakeclaude.py; with a real API key you can pass anthropic.Anthropic() instead.
"""

import random
import time

import anthropic

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# US dollars per million tokens. Prices change: check https://www.anthropic.com/pricing
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


# 1. The text of a Message: the concatenation of the .text of its "text" blocks (ignore
#    every other block type).
def reply_text(message):
    raise NotImplementedError


# 2. One question, one answer: client.messages.create with model=MODEL, max_tokens,
#    messages=[a single user message], output_config={"effort": effort}, and system=system
#    only if a system prompt is given. Return the reply text.
def ask(client, question, system=None, max_tokens=1024, effort="low"):
    raise NotImplementedError


# 3. A multi-turn conversation (README section 3). self.messages starts empty.
#    send(text) sends the whole history plus the new user message (and system=... if set),
#    then appends BOTH the user message and {"role": "assistant", "content": reply.content}
#    to self.messages, and returns the reply text. If the call raises, self.messages must be
#    unchanged.
class Conversation:
    def __init__(self, client, system=None, max_tokens=4096):
        raise NotImplementedError

    def send(self, text):
        raise NotImplementedError


# 4. Stream a reply with client.messages.stream(model=MODEL, max_tokens, messages), calling
#    on_text(chunk) for every text chunk. Return the final Message.
def stream_reply(client, messages, on_text, max_tokens=4096):
    raise NotImplementedError


# 5. The cost in US dollars of one call, from its usage (message.usage) and PRICES[model]:
#    input, output, cache-read and cache-write tokens each at their price per million.
#    cache_read_input_tokens / cache_creation_input_tokens may be None (count them as 0).
def cost_usd(usage, model=MODEL):
    raise NotImplementedError


# 6. Turn an SDK exception into an LLMError(kind, retryable):
#    AuthenticationError -> ("auth", False); BadRequestError -> ("bad_request", False);
#    RateLimitError -> ("rate_limit", True); any other APIStatusError with status >= 500 ->
#    ("server", True), below 500 -> ("client", False); APIConnectionError ->
#    ("connection", True). Anything else: raise TypeError. Check the most specific classes first.
class LLMError(Exception):
    def __init__(self, kind, retryable, message=""):
        super().__init__(message or kind)
        self.kind, self.retryable = kind, retryable


def classify_error(error):
    raise NotImplementedError


# 7. Exponential backoff with jitter: min(max_delay, base_delay * 2 ** attempt) times a factor
#    0.5 + 0.5 * rng.random().
def backoff_delay(attempt, base_delay=1.0, max_delay=30.0, rng=random):
    raise NotImplementedError


# 8. Call fn() up to max_attempts times. On an anthropic.APIError, classify it: if it isn't
#    retryable, or this was the last attempt, raise the LLMError (from the original error).
#    Otherwise sleep(backoff_delay(attempt, ...)) before the next attempt, but for a
#    RateLimitError with a "retry-after" header, sleep at least that many seconds.
#    Return fn()'s result when it succeeds.
def call_with_retries(fn, max_attempts=4, base_delay=1.0, max_delay=30.0, sleep=time.sleep, rng=random):
    raise NotImplementedError


# 9. Ask with server-side refusal fallbacks (README section 8): client.beta.messages.create
#    with model=MODEL, max_tokens, one user message, output_config={"effort": effort},
#    betas=[FALLBACK_BETA], fallbacks="default". Return None if the final stop_reason is
#    "refusal", otherwise the reply text.
def ask_with_fallback(client, question, max_tokens=4096, effort="medium"):
    raise NotImplementedError


# 10. The number of input tokens these messages would use, from
#     client.messages.count_tokens(model=MODEL, messages=..., system=... if given).
def count_tokens(client, messages, system=None):
    raise NotImplementedError
