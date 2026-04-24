#!/usr/bin/env python3
"""Test script for Qwen3-0.6B running on the AX650N via axllm serve.

Start the server first:
    axllm serve ~/qwen3-0.6b/Qwen3-0.6B/

Then run this script:
    python3 test_qwen3.py
"""

import argparse
import time
import requests

DEFAULT_BASE = "http://127.0.0.1:8000/v1"
MODEL = "AXERA-TECH/Qwen3-0.6B"


def chat(base_url: str, messages: list[dict], stream: bool = False) -> str:
    """Send a chat completion request and return the assistant reply."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }

    if stream:
        full = []
        t0 = time.perf_counter()
        first_token_time = None
        with requests.post(url, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full.append(content)
                    print(content, end="", flush=True)
        print()
        elapsed = time.perf_counter() - t0
        ttft = (first_token_time - t0) if first_token_time else elapsed
        text = "".join(full)
        tokens_approx = len(text.split())
        print(f"\n--- stream stats: TTFT={ttft:.2f}s | total={elapsed:.2f}s | ~{tokens_approx} words ---")
        return text
    else:
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=60)
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        result = r.json()
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(text)
        print(f"\n--- non-stream stats: {elapsed:.2f}s", end="")
        if usage:
            total = usage.get("completion_tokens", 0)
            if total and elapsed > 0:
                print(f" | {total} tokens | {total/elapsed:.1f} tok/s", end="")
        print(" ---")
        return text


def run_tests(base_url: str, stream: bool):
    tests = [
        {
            "name": "Basic greeting",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        },
        {
            "name": "Simple reasoning",
            "messages": [{"role": "user", "content": "What is 23 + 19?"}],
        },
        {
            "name": "Knowledge check",
            "messages": [
                {"role": "user", "content": "What is the capital of France? Reply in one sentence."}
            ],
        },
        {
            "name": "Multi-turn conversation",
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
        },
        {
            "name": "Code generation",
            "messages": [
                {"role": "user", "content": "Write a Python function that returns the factorial of n. Keep it short."}
            ],
        },
    ]

    print(f"Running {len(tests)} tests against {base_url}")
    print(f"Model: {MODEL}")
    print(f"Streaming: {stream}")
    print("=" * 60)

    passed = 0
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test['name']}")
        print("-" * 40)
        try:
            reply = chat(base_url, test["messages"], stream=stream)
            if reply.strip():
                print("=> PASS")
                passed += 1
            else:
                print("=> FAIL (empty response)")
        except Exception as e:
            print(f"=> FAIL ({e})")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")


def check_server(base_url: str) -> bool:
    """Check if the server is reachable."""
    try:
        r = requests.get(f"{base_url}/models", timeout=5)
        r.raise_for_status()
        print(f"Server is up: {base_url}")
        models = r.json().get("data", [])
        if models:
            print(f"Available models: {', '.join(m['id'] for m in models)}")
        return True
    except requests.ConnectionError:
        print(f"Cannot connect to {base_url}")
        print("Make sure the server is running: axllm serve ~/qwen3-0.6b/Qwen3-0.6B/")
        return False
    except Exception as e:
        print(f"Server check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-0.6B on AX650N")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument("--stream", action="store_true", help="Use streaming responses")
    parser.add_argument("--prompt", type=str, help="Single prompt instead of running all tests")
    args = parser.parse_args()

    if not check_server(args.base_url):
        return

    print()
    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        chat(args.base_url, messages, stream=args.stream)
    else:
        run_tests(args.base_url, stream=args.stream)


if __name__ == "__main__":
    main()
