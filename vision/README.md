# Real-time NPU vision scripts

Camera → NPU pipelines using [pyaxengine](https://github.com/AXERA-TECH/pyaxengine) and the
YOLO11x `.axmodel` files from M5Stack's `axcl_demo.zip` (expected at `/home/dom/axcl_demo/`).

- `pose_rt.py` — YOLO11x-pose, 17-keypoint skeletons, 22 FPS. `--image` / `--bench` / `--stream` (MJPEG on :8080).
- `detect_rt.py` — YOLO11x detection, 80 COCO classes, 18 FPS. Same modes.
- `watchdog.py` — motion gate → YOLO confirm → snapshot → Qwen2.5-VL description logged to `~/watchdog_captures/`. Needs the VL model from [`../qwen2.5-vl-3b/RUN_QWEN2_5_VL.md`](../qwen2.5-vl-3b/RUN_QWEN2_5_VL.md) unless run with `--no-vlm`.

Usage details for all three are in [`../FIELD_GUIDE.md`](../FIELD_GUIDE.md).

## Environment setup

The scripts need a venv with `axengine`, plus the system `cv2` and `picamera2`:

```bash
python3 -m venv --system-site-packages venv   # reuses system cv2 + picamera2
./venv/bin/pip install ./axengine-0.1.3-py3-none-any.whl   # from pyaxengine releases
./venv/bin/pip install "numpy<2"              # MUST come after the wheel
```

**The numpy trap:** the axengine wheel pulls numpy 2.x, which breaks Pi OS's system cv2
(compiled against numpy 1.x) with `_ARRAY_API not found`. Pin `numpy<2` afterwards; the
resulting `ml-dtypes` pip warning is harmless.

`watchdog.py` imports from `detect_rt.py` — keep them together (on the Pi it looks for it
in `/home/dom/pose-rt/`).
