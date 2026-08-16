# LLM-8850 Field Guide

User guide for the Python demo scripts on this Pi 5 + M5Stack LLM-8850 (AX8850, 24 TOPS).
All commands verified working, August 2026.

## Pick a script

| You want to…                    | Run                                              | Speed      |
|---------------------------------|--------------------------------------------------|------------|
| Chat with a local LLM           | `python3 qwen3_chat.py`                          | ~10 tok/s  |
| Take a photo and ask about it   | `python3 photo_chat.py`                          | ~6 tok/s   |
| Live skeleton tracking          | `pose-rt/venv/bin/python pose-rt/pose_rt.py --stream`   | 22 FPS |
| Detect 80 object types live     | `pose-rt/venv/bin/python pose-rt/detect_rt.py --stream` | 18 FPS |
| Guard a room, log who appears   | `pose-rt/venv/bin/python watchdog.py`            | continuous |

The camera/NPU scripts need the pose-rt venv python (`axengine` + `cv2` + `picamera2`).
The two chat scripts run on plain `python3`, no dependencies.

**The one rule that matters:** the card holds one language model at a time
(Qwen3 text ≈ 2.5 GB, Qwen2.5-VL ≈ 4.8 GB, card total 7 GB). Each chat script stops the
other's server and loads its own — switching costs a 1–2 min load, then stays warm.
YOLO models (~60 MB) fit alongside either.

Ports: **8000** text LLM API · **8001** vision LLM API · **8080** camera MJPEG stream.

---

## qwen3_chat.py — talk to a local LLM

Streams answers from Qwen3-1.7B on the card, fully offline. First run starts the server
(~1 min); it stays warm afterwards, even across script restarts.

```bash
python3 qwen3_chat.py                        # interactive chat
python3 qwen3_chat.py "why is the sky blue?" # one-shot answer
```

In-chat: `/think` toggles reasoning mode (deliberates in gray before answering — slower,
better on hard questions; off by default), `/new` clears history, `/quit` exits.

The server is OpenAI-compatible: `http://<pi-ip>:8000/v1/chat/completions` works from any
machine on the network. One request at a time.

## photo_chat.py — photograph, describe, interrogate

Takes a photo (1.5 s exposure settle — aim first), Qwen2.5-VL describes it, then you ask
follow-up questions about the image.

```bash
python3 photo_chat.py             # camera photo, then Q&A
python3 photo_chat.py holiday.jpg # use an existing image
```

In-chat: `/photo` retakes and re-describes, `/quit` exits.
Latest capture: `/home/dom/photo_chat_latest.jpg`. Each question is answered against the
image alone (no chat memory) — phrase questions self-contained.

## pose_rt.py — live skeleton tracking (in /home/dom/pose-rt/)

YOLO11x-pose on the NPU, 17-keypoint skeletons on everyone in view. 22.3 FPS.

```bash
cd /home/dom/pose-rt
venv/bin/python pose_rt.py --stream        # watch at http://<pi-ip>:8080/
venv/bin/python pose_rt.py --image me.jpg  # annotate one image
venv/bin/python pose_rt.py --bench         # FPS breakdown
```

Options: `--conf 0.4` threshold, `--width/--height` capture size (default 640×480).

## detect_rt.py — live object detection (in /home/dom/pose-rt/)

Same pipeline, 80 COCO classes (people, pets, vehicles, furniture, food, electronics…),
labeled boxes with per-class colors. 18.2 FPS. Same modes/options as pose_rt.py.
`--image` mode also prints a ranked list of findings with confidence + coordinates.

## watchdog.py — security camera that writes reports

Motion check (CPU) → YOLO11x confirms what moved (NPU) → annotated snapshot saved →
Qwen2.5-VL writes a plain-English line in the event log:

```
[20260816_164824] motion 2% | person 81% | ~/watchdog_captures/20260816_164824.jpg
  A person is wearing glasses and looking at the ceiling.
```

```bash
/home/dom/pose-rt/venv/bin/python watchdog.py                    # alert on people
/home/dom/pose-rt/venv/bin/python watchdog.py --classes person,cat,dog
/home/dom/pose-rt/venv/bin/python watchdog.py --no-vlm           # detection only
```

Options: `--classes person` (or `any`) · `--motion 0.02` sensitivity (lower = more
sensitive) · `--conf 0.5` · `--cooldown 30` s between events · `--no-vlm` skips
descriptions and the 2-min vision model load.

Events accumulate in `~/watchdog_captures/` (snapshots + `watchdog.log`) — prune
occasionally. While a description is written (~15–30 s) watching pauses; `--no-vlm`
never pauses. Runs until Ctrl-C; use `tmux` for always-on.

---

## Troubleshooting

- **Server won't start / responses hang** — stale servers fighting over card memory:
  `pkill -f "axllm serve"`, wait a few seconds, rerun.
- **Is the card alive?** — `axcl-smi` shows temp, memory, processes.
- **Camera "in use"** — one camera consumer at a time; stop the other script.
- **Broke after a kernel update** — DKMS rebuild + dangling `/usr/lib/axcl/*.so` links;
  see memory notes for the one-line fixes.
- **First response after switching chat scripts is slow** — the 1–2 min model swap. Normal.
