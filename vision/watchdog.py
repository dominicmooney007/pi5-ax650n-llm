#!/usr/bin/env python3
"""Security watchdog on the LLM-8850 NPU.

Pipeline: Pi camera watches for motion (cheap frame differencing on the CPU)
-> YOLO11x on the NPU confirms what moved -> snapshot saved -> Qwen2.5-VL
writes a plain-English description to the event log.

Usage (needs the pose-rt venv for camera + NPU libs):
  /home/dom/pose-rt/venv/bin/python watchdog.py               # watch for people
  /home/dom/pose-rt/venv/bin/python watchdog.py --classes person,cat,dog
  /home/dom/pose-rt/venv/bin/python watchdog.py --no-vlm      # skip descriptions

Events land in ~/watchdog_captures/ (snapshots + watchdog.log). Ctrl-C stops.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

sys.path.insert(0, "/home/dom/pose-rt")
import cv2
import numpy as np
from detect_rt import COCO, Detector, draw

VL_MODEL_DIR = ("/home/dom/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/"
                "Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512")
VL_MODEL_ID = "AXERA-TECH/Qwen2.5-VL-3B-Instruct"
VL_PORT = 8001
VL_BASE = f"http://127.0.0.1:{VL_PORT}"
OUT_DIR = os.path.expanduser("~/watchdog_captures")

GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------- Qwen2.5-VL over the axllm HTTP API ----------

def vl_served() -> bool:
    try:
        with urllib.request.urlopen(f"{VL_BASE}/v1/models", timeout=2) as r:
            return json.load(r)["data"][0]["id"] == VL_MODEL_ID
    except (urllib.error.URLError, OSError, KeyError, IndexError):
        return False


def ensure_vl_server():
    if vl_served():
        return
    # One model at a time on the card: stop any other axllm server first.
    subprocess.run(["pkill", "-f", "axllm serve"], capture_output=True)
    time.sleep(3)
    print(f"{GRAY}Loading vision model (~2 min, one-off)...{RESET}")
    subprocess.Popen(
        ["axllm", "serve", VL_MODEL_DIR, "--port", str(VL_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(180):
        time.sleep(2)
        if vl_served():
            print(f"{GRAY}Vision model ready.{RESET}")
            return
    sys.exit(f"VL server failed to start. Try: axllm serve {VL_MODEL_DIR}")


def describe(jpeg_path: str) -> str:
    with open(jpeg_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({
        "model": VL_MODEL_ID,
        "max_tokens": 128,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "This is a frame from a security camera. Describe "
                         "what is happening in one or two sentences."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        f"{VL_BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


# ---------- Motion detection ----------

def motion_score(prev_small, small):
    """Fraction of pixels that changed noticeably between two frames."""
    diff = cv2.absdiff(prev_small, small)
    return float((diff > 25).mean())


def to_small(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (160, 120))
    return cv2.GaussianBlur(g, (7, 7), 0)


# ---------- Main loop ----------

def main():
    sys.stdout.reconfigure(line_buffering=True)  # live output even when piped
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="person",
                    help="comma-separated COCO classes to alert on, or 'any'")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--motion", type=float, default=0.02,
                    help="fraction of frame that must change to trigger")
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="seconds between logged events")
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip Qwen2.5-VL descriptions (detection only)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    watch = None if args.classes == "any" else {
        c.strip() for c in args.classes.split(",")}
    if watch:
        unknown = watch - set(COCO)
        if unknown:
            sys.exit(f"unknown classes: {', '.join(unknown)} (see COCO list)")

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "watchdog.log")

    if not args.no_vlm:
        ensure_vl_server()

    det = Detector(conf=args.conf)
    print(f"{GRAY}Detector loaded in {det.load_time:.2f}s{RESET}")

    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    cam.start()
    time.sleep(1.0)

    targets = "anything" if watch is None else ", ".join(sorted(watch))
    print(f"{BOLD}Watching for: {targets}{RESET}  "
          f"(motion>{args.motion:.0%}, conf>{args.conf:.0%}, "
          f"cooldown {args.cooldown:.0f}s)\nEvents -> {OUT_DIR}  Ctrl-C to stop")

    prev_small = to_small(cam.capture_array())
    last_event = 0.0
    try:
        while True:
            frame = cam.capture_array()
            small = to_small(frame)
            score = motion_score(prev_small, small)
            prev_small = small

            if score < args.motion or time.time() - last_event < args.cooldown:
                time.sleep(0.1)
                continue

            boxes, scores, classes = det(frame)
            names = [COCO[int(c)] for c in classes]
            hits = [(b, s, n) for b, s, n in zip(boxes, scores, names)
                    if watch is None or n in watch]
            if not hits:
                continue

            last_event = time.time()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = os.path.join(OUT_DIR, f"{stamp}.jpg")
            annotated = draw(frame.copy(), boxes, scores, classes)
            cv2.imwrite(snap, annotated)

            found = ", ".join(f"{n} {s:.0%}" for _, s, n in
                              sorted(hits, key=lambda h: -h[1]))
            line = f"[{stamp}] motion {score:.0%} | {found} | {snap}"
            print(f"\n{BOLD}{line}{RESET}")

            if not args.no_vlm:
                print(f"{GRAY}Asking Qwen2.5-VL what it sees...{RESET}")
                try:
                    desc = describe(snap)
                except (urllib.error.URLError, OSError) as e:
                    desc = f"(description failed: {e})"
                print(f"  {desc}")
                line += f"\n  {desc}"

            with open(log_path, "a") as f:
                f.write(line + "\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
