#!/usr/bin/env python3
"""Capture frames from a USB camera and describe them with Qwen2.5-VL-3B via axllm serve.

Usage:
    # start the server in another terminal first:
    #   axllm serve ~/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/
    python3 camera_describe.py
    python3 camera_describe.py --camera 1 --interval 0 --prompt "What is the person doing?"
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

import cv2


def grab_frame(cap):
    # UVC cameras often hand back a stale buffered frame on the first read after idle.
    # Drain a few to get a fresh one.
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        ok, frame = cap.read()
    return frame if ok else None


def describe(url, model, prompt, jpeg_bytes, max_tokens, timeout):
    img_b64 = base64.b64encode(jpeg_bytes).decode()
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0,
                    help="V4L2 device index, e.g. 0 for /dev/video0")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="Seconds to pause between frames (0 = back-to-back)")
    ap.add_argument("--prompt", default="Describe what you see in one sentence.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="AXERA-TECH/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--save-frames", metavar="DIR",
                    help="If set, write each captured JPEG to this directory")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Could not open /dev/video{args.camera}. "
                 "Plug in a UVC USB camera or pick a different --camera index.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if args.save_frames:
        import os
        os.makedirs(args.save_frames, exist_ok=True)

    print(f"Capturing from /dev/video{args.camera} at "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. Ctrl-C to stop.",
          file=sys.stderr, flush=True)

    n = 0
    try:
        while True:
            frame = grab_frame(cap)
            if frame is None:
                print("[warn] dropped frame", file=sys.stderr, flush=True)
                time.sleep(0.2)
                continue

            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                print("[warn] jpeg encode failed", file=sys.stderr, flush=True)
                continue
            jpeg_bytes = buf.tobytes()

            if args.save_frames:
                import os
                with open(os.path.join(args.save_frames,
                                       f"frame_{n:06d}.jpg"), "wb") as f:
                    f.write(jpeg_bytes)

            t0 = time.time()
            try:
                text = describe(args.url, args.model, args.prompt,
                                jpeg_bytes, args.max_tokens, args.timeout)
            except urllib.error.URLError as e:
                sys.exit(f"Could not reach {args.url}: {e}. "
                         "Is `axllm serve` running?")
            dt = time.time() - t0

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] ({dt:4.1f}s) {text}", flush=True)

            n += 1
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
