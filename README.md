# pi5-ax650n-llm

Notes, configs, and scripts for running Qwen LLMs on a Raspberry Pi 5 with an **M5Stack LLM-8850** (AX650N SoC) M.2 accelerator.

## Contents

- [`qwen3-0.6b/RUN_QWEN3.md`](qwen3-0.6b/RUN_QWEN3.md) — end-to-end setup for a small text-only LLM: installing the AXCL driver, building `axllm`, downloading the model, running interactively or as an OpenAI-compatible HTTP server.
- [`qwen2.5-vl-3b/RUN_QWEN2_5_VL.md`](qwen2.5-vl-3b/RUN_QWEN2_5_VL.md) — follow-up guide for a vision-language model. Documents AXERA's empty-`config.json` gotcha and the hand-written config that makes `axllm` work with it.
- [`qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/`](qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/) — the hand-written `config.json` and `post_config.json` for Qwen2.5-VL-3B. Drop them straight into the equivalently-named weights directory in the AXERA HuggingFace clone.
- [`qwen2.5-vl-3b/camera_describe.py`](qwen2.5-vl-3b/camera_describe.py) — Python script that grabs frames from a USB webcam and pipes them through the vision model via the OpenAI-compatible `axllm serve` endpoint. Prints a rolling description to stdout.
- [`FIELD_GUIDE.md`](FIELD_GUIDE.md) — operator's guide to all the demo scripts below: commands, options, measured speeds, and troubleshooting.
- [`qwen3-1.7b/qwen3_chat.py`](qwen3-1.7b/qwen3_chat.py) — stdlib-only chat client for Qwen3-1.7B. Auto-starts `axllm serve`, streams replies, `/think` toggles reasoning mode.
- [`qwen2.5-vl-3b/photo_chat.py`](qwen2.5-vl-3b/photo_chat.py) — takes a photo with the Pi camera, has Qwen2.5-VL describe it, then answers follow-up questions about the image.
- [`vision/`](vision/) — real-time camera→NPU pipelines via pyaxengine: `pose_rt.py` (22 FPS skeletons), `detect_rt.py` (18 FPS, 80 classes), and `watchdog.py` (motion → YOLO confirm → snapshot → VL description log). See its README for the venv setup and the numpy pin it needs.

## What this repo is not

It does **not** ship model weights. The `.axmodel` layer files and embedding bins are several GB each and live on HuggingFace — grab them from [huggingface.co/AXERA-TECH](https://huggingface.co/AXERA-TECH) per the instructions in each guide.

## Hardware

- Raspberry Pi 5 (8 GB)
- M5Stack LLM-8850 M.2 accelerator, seated on the official Raspberry Pi M.2 HAT+
- AXCL host driver v3.6.4
