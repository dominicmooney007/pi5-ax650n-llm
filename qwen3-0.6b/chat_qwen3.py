#!/usr/bin/env python3
"""Interactive chat with Qwen3-0.6B running on the AX650N via axllm serve.

Start the server first:
    axllm serve ~/qwen3-0.6b/Qwen3-0.6B/

Then run:
    python3 chat_qwen3.py
"""

import argparse
import json
import requests

DEFAULT_BASE = "http://127.0.0.1:8000/v1"
MODEL = "AXERA-TECH/Qwen3-0.6B"


def stream_reply(base_url: str, messages: list[dict]) -> str:
    """Send messages and stream the response token by token."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
    }
    full = []
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            if content:
                full.append(content)
                print(content, end="", flush=True)
    print()
    return "".join(full)


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3-0.6B on AX650N")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    args = parser.parse_args()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    # Check server is reachable
    try:
        r = requests.get(f"{args.base_url}/models", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        print(f"Cannot connect to {args.base_url}")
        print("Start the server first: axllm serve ~/qwen3-0.6b/Qwen3-0.6B/")
        return

    print("Qwen3-0.6B Chat (type /quit to exit, /reset to clear history)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if user_input == "/reset":
            messages = messages[:1] if args.system else []
            print("(conversation cleared)")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nQwen3: ", end="", flush=True)
        try:
            reply = stream_reply(args.base_url, messages)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[error: {e}]")
            messages.pop()  # remove the failed user message


if __name__ == "__main__":
    main()
