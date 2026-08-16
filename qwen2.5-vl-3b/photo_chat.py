#!/usr/bin/env python3
"""Take a photo with the Pi camera, describe it with Qwen2.5-VL on the
LLM-8850 NPU, then ask follow-up questions about it.

Usage:
  python3 photo_chat.py               # capture, describe, then Q&A loop
  python3 photo_chat.py photo.jpg     # skip the camera, use an existing image

In-chat commands: /photo (take a new picture), /quit
Starts `axllm serve` for the VL model automatically (first load ~90 s).
"""

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = ("/home/dom/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/"
             "Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512")
MODEL_ID = "AXERA-TECH/Qwen2.5-VL-3B-Instruct"
PORT = 8001  # 8000 is the Qwen3 text model's port
BASE = f"http://127.0.0.1:{PORT}"
PHOTO = "/home/dom/photo_chat_latest.jpg"

GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def served_model() -> str | None:
    """Return the model id being served on our port, or None."""
    try:
        with urllib.request.urlopen(f"{BASE}/v1/models", timeout=2) as resp:
            return json.load(resp)["data"][0]["id"]
    except (urllib.error.URLError, OSError, KeyError, IndexError):
        return None


def ensure_server():
    if served_model() == MODEL_ID:
        return
    # The card fits only one model at a time (VL needs ~4.8 GB of 7 GB CMM),
    # so stop any other axllm server first. qwen3_chat.py restarts its own.
    subprocess.run(["pkill", "-f", "axllm serve"], capture_output=True)
    time.sleep(3)
    print(f"{GRAY}Starting vision model server (first load takes ~2 min)...{RESET}")
    subprocess.Popen(
        ["axllm", "serve", MODEL_DIR, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # keep serving after this script exits
    )
    for _ in range(180):
        time.sleep(2)
        if served_model() == MODEL_ID:
            print(f"{GRAY}Server ready.{RESET}")
            return
    sys.exit(f"Server failed to start. Try manually: axllm serve {MODEL_DIR}")


def take_photo(path: str = PHOTO) -> str:
    print(f"{GRAY}Taking photo...{RESET}")
    subprocess.run(
        ["rpicam-still", "-n", "-t", "1500",
         "--width", "1280", "--height", "720", "-o", path],
        check=True, capture_output=True,
    )
    return path


def ask(image_b64: str, question: str) -> str:
    """Send image + question, stream the answer, return the full text."""
    body = json.dumps({
        "model": MODEL_ID,
        "stream": True,
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    text = ""
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            if not text:
                piece = piece.lstrip()
            text += piece
            print(piece, end="", flush=True)
    print()
    return text


def load_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    ensure_server()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"{GRAY}Using image: {path}{RESET}")
    else:
        path = take_photo()
    image = load_image(path)

    print(f"\n{BOLD}Description:{RESET}")
    ask(image, "Describe this image in a few sentences.")

    print(f"\n{GRAY}Ask questions about the photo. /photo retakes, /quit exits.{RESET}")
    while True:
        try:
            q = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q == "/photo":
            image = load_image(take_photo())
            print(f"\n{BOLD}Description:{RESET}")
            ask(image, "Describe this image in a few sentences.")
            continue
        try:
            ask(image, q)
        except (urllib.error.URLError, OSError) as e:
            print(f"Request failed ({e}); is the server still running?")


if __name__ == "__main__":
    main()
