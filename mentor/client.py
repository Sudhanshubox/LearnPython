"""Talks to Claude: streams replies and runs interactive terminal sessions."""

import anthropic

from . import config, progress, prompts

EXIT_WORDS = {"exit", "quit", "bye", "/exit", "/quit"}

NO_KEY = (
    "No Anthropic credentials found. Get an API key at "
    "https://console.anthropic.com and run:\n  export ANTHROPIC_API_KEY=sk-ant-..."
)


class MentorError(Exception):
    """A problem talking to the API, with a message fit to show the learner."""


def system_prompt():
    learner = progress.summary(progress.load())
    return prompts.SYSTEM + "\n--- Learner context ---\n" + learner


def ask(client, system, messages, on_text):
    """Stream one reply, calling on_text(chunk) as text arrives. Returns the final message."""
    try:
        with client.beta.messages.stream(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": config.EFFORT},
            cache_control={"type": "ephemeral"},
            betas=[config.FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                on_text(text)
            message = stream.get_final_message()
    except TypeError as e:
        # The SDK raises TypeError when it can't find any credentials.
        if "authentication" in str(e):
            raise MentorError(NO_KEY) from e
        raise
    except anthropic.AuthenticationError as e:
        raise MentorError("Your API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise MentorError("Rate limited by the API. Wait a minute and try again.") from e
    except anthropic.APIStatusError as e:
        raise MentorError(f"API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise MentorError("Couldn't reach the API. Check your internet connection.") from e
    if message.stop_reason == "refusal":
        on_text("\n\n[The mentor declined to answer that. Try rephrasing the question.]")
    elif message.stop_reason == "max_tokens":
        on_text("\n\n[Reply was cut off. Say 'continue' to get the rest.]")
    return message


def text_of_blocks(blocks):
    return "".join(b.text for b in blocks if b.type == "text")


def _print(text):
    print(text, end="", flush=True)


def session(opening, interactive=True):
    """Send `opening` as the first user turn, then keep chatting in the terminal until the user exits.

    Returns the full conversation so callers can inspect the replies.
    """
    client = anthropic.Anthropic()
    system = system_prompt()
    messages = [{"role": "user", "content": opening}]
    while True:
        print("\nmentor> ", end="")
        try:
            reply = ask(client, system, messages, _print)
        except MentorError as e:
            raise SystemExit(f"\n{e}")
        print()
        # Keep the reply's content blocks as-is so the history stays append-only.
        messages.append({"role": "assistant", "content": reply.content})
        if not interactive:
            return messages
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return messages
        if user.lower() in EXIT_WORDS:
            return messages
        if not user:
            user = "continue"
        messages.append({"role": "user", "content": user})
