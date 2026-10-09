"""Talks to Claude: streams replies and runs interactive chat sessions."""

import anthropic

from . import config, progress, prompts

EXIT_WORDS = {"exit", "quit", "bye", "/exit", "/quit"}


NO_KEY = (
    "\nNo Anthropic credentials found. Get an API key at "
    "https://console.anthropic.com and run:\n  export ANTHROPIC_API_KEY=sk-ant-..."
)


def _system():
    learner = progress.summary(progress.load())
    return prompts.SYSTEM + "\n--- Learner context ---\n" + learner


def ask(client, system, messages):
    """Stream one reply to the terminal. Returns the final message."""
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
                print(text, end="", flush=True)
            message = stream.get_final_message()
    except TypeError as e:
        # The SDK raises TypeError when it can't find any credentials.
        if "authentication" in str(e):
            raise SystemExit(NO_KEY)
        raise
    except anthropic.AuthenticationError:
        raise SystemExit("\nYour API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise SystemExit("\nRate limited by the API. Wait a minute and try again.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"\nAPI error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise SystemExit("\nCouldn't reach the API. Check your internet connection.")
    print()
    if message.stop_reason == "refusal":
        print("[The mentor declined to answer that. Try rephrasing the question.]")
    elif message.stop_reason == "max_tokens":
        print("[Reply was cut off. Say 'continue' to get the rest.]")
    return message


def text_of_blocks(blocks):
    return "".join(b.text for b in blocks if b.type == "text")


def session(opening, interactive=True):
    """Send `opening` as the first user turn, then keep chatting until the user exits.

    Returns the full conversation so callers can inspect the replies.
    """
    client = anthropic.Anthropic()
    system = _system()
    messages = [{"role": "user", "content": opening}]
    while True:
        print("\nmentor> ", end="")
        reply = ask(client, system, messages)
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
