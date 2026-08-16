#!/usr/bin/env python3
"""Real-time YOLO11x-pose on the LLM-8850 (AX8850) NPU.

The model is loaded ONCE and reused, which is the whole point: the M5Stack
axcl_yolo11_pose binary reloads 60MB per invocation (~1.6s), capping a shell
loop at ~0.5 FPS. Here load is a one-off ~1.3s and each frame costs ~25ms.

Modes:
  --image FILE     run on one image, write annotated output (validation)
  --bench [N]      camera -> NPU for N frames, report FPS breakdown
  --stream         camera -> NPU -> MJPEG on http://<pi>:8080/
"""
import argparse
import sys
import time

import cv2
import numpy as np

MODEL = "/home/dom/axcl_demo/yolo11x-pose.axmodel"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
REG_MAX = 16
NUM_KPT = 17

# COCO 17-keypoint skeleton.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=INPUT_SIZE):
    """Resize preserving aspect ratio, pad to square. Returns img + inverse params."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def decode(outputs, conf_thres):
    """Decode YOLO11-pose heads (DFL boxes + keypoints) at all three scales."""
    boxes, scores, kpts = [], [], []

    for stride in STRIDES:
        box_raw = outputs[("box", stride)]     # (g, g, 65)
        kpt_raw = outputs[("kpt", stride)]     # (g, g, 51)
        g = box_raw.shape[0]

        cls = sigmoid(box_raw[..., 4 * REG_MAX])
        gy, gx = np.nonzero(cls > conf_thres)
        if gy.size == 0:
            continue

        # DFL: 4 sides x 16 bins -> expected distance, in grid units.
        dfl = box_raw[gy, gx, :4 * REG_MAX].reshape(-1, 4, REG_MAX)
        dfl = dfl - dfl.max(axis=-1, keepdims=True)
        prob = np.exp(dfl)
        prob /= prob.sum(axis=-1, keepdims=True)
        dist = (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)  # (N,4) l,t,r,b

        # Anchor points sit at cell centres (grid + 0.5), as in Ultralytics.
        ax, ay = gx + 0.5, gy + 0.5
        x1 = (ax - dist[:, 0]) * stride
        y1 = (ay - dist[:, 1]) * stride
        x2 = (ax + dist[:, 2]) * stride
        y2 = (ay + dist[:, 3]) * stride
        boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        scores.append(cls[gy, gx])

        k = kpt_raw[gy, gx].reshape(-1, NUM_KPT, 3)
        kx = (k[..., 0] * 2.0 + gx[:, None]) * stride
        ky = (k[..., 1] * 2.0 + gy[:, None]) * stride
        kpts.append(np.stack([kx, ky, sigmoid(k[..., 2])], axis=-1))

    if not boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, NUM_KPT, 3))
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpts)


def nms(boxes, scores, kpts, iou_thres):
    if len(boxes) == 0:
        return boxes, scores, kpts
    # cv2's NMS wants xywh.
    wh = np.stack([boxes[:, 0], boxes[:, 1],
                   boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    keep = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), 0.0, iou_thres)
    if len(keep) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, NUM_KPT, 3))
    keep = np.array(keep).flatten()
    return boxes[keep], scores[keep], kpts[keep]


class PoseDetector:
    def __init__(self, model=MODEL, conf=0.4, iou=0.45):
        import axengine as axe
        self.conf, self.iou = conf, iou
        t0 = time.time()
        self.sess = axe.InferenceSession(model)
        self.load_time = time.time() - t0
        self.input_name = self.sess.get_inputs()[0].name
        # Map output tensors to (kind, stride) by their channel count and grid size.
        self.out_names = []
        for o in self.sess.get_outputs():
            _, g, _, c = o.shape
            kind = "box" if c == 4 * REG_MAX + 1 else "kpt"
            self.out_names.append((o.name, kind, INPUT_SIZE // g))

    def __call__(self, bgr):
        """Returns (boxes, scores, kpts) in original-image coordinates."""
        padded, scale, dx, dy = letterbox(bgr)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        raw = self.sess.run(None, {self.input_name: rgb[None, ...]})

        outputs = {}
        for arr, (_, kind, stride) in zip(raw, self.out_names):
            outputs[(kind, stride)] = arr[0]

        boxes, scores, kpts = decode(outputs, self.conf)
        boxes, scores, kpts = nms(boxes, scores, kpts, self.iou)

        # Undo the letterbox.
        if len(boxes):
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
            kpts[..., 0] = (kpts[..., 0] - dx) / scale
            kpts[..., 1] = (kpts[..., 1] - dy) / scale
        return boxes, scores, kpts


def draw(img, boxes, scores, kpts, kpt_thres=0.5):
    for box, score, kp in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(img, f"person {score:.0%}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for a, b in SKELETON:
            if kp[a, 2] > kpt_thres and kp[b, 2] > kpt_thres:
                cv2.line(img, tuple(kp[a, :2].astype(int)),
                         tuple(kp[b, :2].astype(int)), (0, 255, 255), 2)
        for x, y, c in kp:
            if c > kpt_thres:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    return img


def open_camera(width, height):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"})
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)  # let AE/AWB settle
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
    ap.add_argument("-o", "--out", default="pose_out.jpg")
    args = ap.parse_args()

    det = PoseDetector(conf=args.conf)
    print(f"model loaded in {det.load_time:.2f}s", file=sys.stderr)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read {args.image}")
        t0 = time.time()
        boxes, scores, kpts = det(img)
        print(f"inference+post: {(time.time() - t0) * 1000:.1f} ms")
        print(f"detection num: {len(boxes)}")
        for b, s in sorted(zip(boxes, scores), key=lambda z: -z[1]):
            print(f" 0: {s:4.0%}, [{b[0]:5.0f},{b[1]:5.0f},{b[2]:5.0f},{b[3]:5.0f}], person")
        cv2.imwrite(args.out, draw(img, boxes, scores, kpts))
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
                        boxes, scores, kpts = det(frame)
                        draw(frame, boxes, scores, kpts)
                        now = time.time()
                        state["fps"] = 0.9 * state["fps"] + 0.1 / max(now - prev, 1e-6)
                        prev = now
                        cv2.putText(frame, f"{state['fps']:.1f} FPS  {len(boxes)} person",
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
