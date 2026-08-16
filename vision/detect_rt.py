#!/usr/bin/env python3
"""Real-time YOLO11x object detection (80 COCO classes) on the LLM-8850 NPU.

Same pipeline as pose_rt.py: model loads once (~1.3s), each frame ~25ms.

Modes:
  --image FILE     run on one image, write annotated output (validation)
  --bench [N]      camera -> NPU for N frames, report FPS breakdown
  --stream         camera -> NPU -> MJPEG on http://<pi>:8080/

Run with the pose-rt venv: /home/dom/pose-rt/venv/bin/python detect_rt.py
"""
import argparse
import sys
import time

import cv2
import numpy as np

MODEL = "/home/dom/axcl_demo/yolo11x.axmodel"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
REG_MAX = 16

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=INPUT_SIZE):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def decode(outputs, conf_thres):
    """Decode YOLO11 detection heads: 64 DFL box channels + 80 class scores."""
    boxes, scores, classes = [], [], []

    for stride in STRIDES:
        raw = outputs[stride]                      # (g, g, 144)
        cls = sigmoid(raw[..., 4 * REG_MAX:])      # (g, g, 80)
        best = cls.max(axis=-1)
        gy, gx = np.nonzero(best > conf_thres)
        if gy.size == 0:
            continue

        dfl = raw[gy, gx, :4 * REG_MAX].reshape(-1, 4, REG_MAX)
        dfl = dfl - dfl.max(axis=-1, keepdims=True)
        prob = np.exp(dfl)
        prob /= prob.sum(axis=-1, keepdims=True)
        dist = (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)

        ax, ay = gx + 0.5, gy + 0.5
        x1 = (ax - dist[:, 0]) * stride
        y1 = (ay - dist[:, 1]) * stride
        x2 = (ax + dist[:, 2]) * stride
        y2 = (ay + dist[:, 3]) * stride
        boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        scores.append(best[gy, gx])
        classes.append(cls[gy, gx].argmax(axis=-1))

    if not boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def nms(boxes, scores, classes, iou_thres):
    if len(boxes) == 0:
        return boxes, scores, classes
    # Offset boxes per class so NMS is class-aware in one pass.
    off = classes.astype(np.float32)[:, None] * INPUT_SIZE * 2
    shifted = boxes + off
    wh = np.stack([shifted[:, 0], shifted[:, 1],
                   shifted[:, 2] - shifted[:, 0],
                   shifted[:, 3] - shifted[:, 1]], axis=1)
    keep = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), 0.0, iou_thres)
    if len(keep) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    keep = np.array(keep).flatten()
    return boxes[keep], scores[keep], classes[keep]


class Detector:
    def __init__(self, model=MODEL, conf=0.4, iou=0.45):
        import axengine as axe
        self.conf, self.iou = conf, iou
        t0 = time.time()
        self.sess = axe.InferenceSession(model)
        self.load_time = time.time() - t0
        self.input_name = self.sess.get_inputs()[0].name
        self.out_strides = [INPUT_SIZE // o.shape[1] for o in self.sess.get_outputs()]

    def __call__(self, bgr):
        """Returns (boxes, scores, class_ids) in original-image coordinates."""
        padded, scale, dx, dy = letterbox(bgr)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        raw = self.sess.run(None, {self.input_name: rgb[None, ...]})

        outputs = {s: arr[0] for arr, s in zip(raw, self.out_strides)}
        boxes, scores, classes = decode(outputs, self.conf)
        boxes, scores, classes = nms(boxes, scores, classes, self.iou)

        if len(boxes):
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
        return boxes, scores, classes


def color_for(cls_id):
    rng = np.random.default_rng(cls_id)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def draw(img, boxes, scores, classes):
    for box, score, cid in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(int)
        col = color_for(int(cid))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{COCO[int(cid)]} {score:.0%}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return img


def open_camera(width, height):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--bench", nargs="?", type=int, const=100)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("-o", "--out", default="detect_out.jpg")
    args = ap.parse_args()

    det = Detector(conf=args.conf)
    print(f"model loaded in {det.load_time:.2f}s", file=sys.stderr)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read {args.image}")
        t0 = time.time()
        boxes, scores, classes = det(img)
        print(f"inference+post: {(time.time() - t0) * 1000:.1f} ms")
        print(f"detections: {len(boxes)}")
        for b, s, c in sorted(zip(boxes, scores, classes), key=lambda z: -z[1]):
            print(f" {COCO[int(c)]:>14}: {s:4.0%} "
                  f"[{b[0]:5.0f},{b[1]:5.0f},{b[2]:5.0f},{b[3]:5.0f}]")
        cv2.imwrite(args.out, draw(img, boxes, scores, classes))
        print(f"wrote {args.out}")
        return

    if args.bench:
        cam = open_camera(args.width, args.height)
        cap_t = inf_t = 0.0
        n = args.bench
        t_start = time.time()
        for _ in range(n):
            t0 = time.time()
            frame = cam.capture_array()
            t1 = time.time()
            det(frame)
            t2 = time.time()
            cap_t += t1 - t0
            inf_t += t2 - t1
        total = time.time() - t_start
        cam.stop()
        print(f"\nframes           : {n}")
        print(f"capture   avg    : {cap_t / n * 1000:6.1f} ms")
        print(f"infer+post avg   : {inf_t / n * 1000:6.1f} ms")
        print(f"end-to-end avg   : {total / n * 1000:6.1f} ms")
        print(f"throughput       : {n / total:6.1f} FPS")
        return

    if args.stream:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        cam = open_camera(args.width, args.height)
        state = {"fps": 0.0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                prev = time.time()
                try:
                    while True:
                        frame = cam.capture_array()
                        boxes, scores, classes = det(frame)
                        draw(frame, boxes, scores, classes)
                        now = time.time()
                        state["fps"] = 0.9 * state["fps"] + 0.1 / max(now - prev, 1e-6)
                        prev = now
                        cv2.putText(frame,
                                    f"{state['fps']:.1f} FPS  {len(boxes)} objects",
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)
                        ok, jpg = cv2.imencode(".jpg", frame)
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        srv = HTTPServer(("0.0.0.0", args.port), Handler)
        print(f"MJPEG stream on http://0.0.0.0:{args.port}/  (Ctrl-C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()
        return

    ap.print_help()


if __name__ == "__main__":
    main()
