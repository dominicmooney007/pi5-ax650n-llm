#!/usr/bin/env python3
"""Chat with Qwen3-1.7B running locally on the LLM-8850 NPU.

Usage:
  python3 qwen3_chat.py               # interactive chat
  python3 qwen3_chat.py "question"    # one-shot answer

Starts `axllm serve` automatically if it isn't already running.
In-chat commands: /new (clear history), /think (toggle reasoning), /quit
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = "/home/dom/qwen3-1.7b/Qwen3-1.7B"
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"

GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_server():
    if server_up():
        return
    # The card fits only one model at a time — stop any other axllm server
    # (e.g. photo_chat.py's vision model) before loading this one.
    subprocess.run(["pkill", "-f", "axllm serve"], capture_output=True)
    time.sleep(3)
    print(f"{GRAY}Starting axllm server (model load takes ~1 min)...{RESET}")
    subprocess.Popen(
        ["axllm", "serve", MODEL_DIR, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # keep serving after this script exits
    )
    for _ in range(120):
        time.sleep(2)
        if server_up():
            print(f"{GRAY}Server ready.{RESET}\n")
            return
    sys.exit("Server failed to start. Try manually: axllm serve " + MODEL_DIR)


def model_name() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as resp:
        return json.load(resp)["data"][0]["id"]


def chat_stream(messages):
    """Stream a reply, printing thinking in gray and the answer normally.

    Returns the full raw reply (including think tags) for the history.
    """
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps({"model": model_name(), "stream": True,
                         "messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
    )
    raw, in_think, printed_think, answer_started = "", False, False, False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            piece = delta.get("content", "")
            raw += piece
            # Route text between <think>...</think> to gray, the rest normal.
            if piece == "<think>":
                in_think = True
            elif piece == "</think>":
                in_think = False
                if printed_think:
                    print(RESET, end="", flush=True)
            elif in_think:
                if piece.strip() and not printed_think:
                    printed_think = True
                    print(f"{GRAY}[thinking] ", end="", flush=True)
                if printed_think:
                    print(f"{GRAY}{piece}{RESET}", end="", flush=True)
            else:
                # Drop leading whitespace after the think block.
                if not answer_started:
                    piece = piece.lstrip()
                    answer_started = bool(piece)
                if piece:
                    print(piece, end="", flush=True)
    print()
    return raw


def main():
    ensure_server()

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    thinking = False

    if len(sys.argv) > 1:  # one-shot mode
        prompt = " ".join(sys.argv[1:])
        history.append({"role": "user", "content": prompt + (" " if thinking else " /no_think")})
        chat_stream(history)
        return

    print(f"{BOLD}Qwen3-1.7B on LLM-8850{RESET}  (/new /think /quit)")
    while True:
        try:
            user = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/new":
            history = history[:1]
            print(f"{GRAY}History cleared.{RESET}")
            continue
        if user == "/think":
            thinking = not thinking
            print(f"{GRAY}Reasoning {'on' if thinking else 'off'}.{RESET}")
            continue

        history.append({"role": "user",
                        "content": user + ("" if thinking else " /no_think")})
        try:
            reply = chat_stream(history)
        except (urllib.error.URLError, OSError) as e:
            print(f"Request failed ({e}); is the server still running?")
            history.pop()
            continue
        # Strip think block from history to keep the context window lean.
        answer = reply.split("</think>")[-1].strip()
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
