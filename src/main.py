# src/main.py
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import yaml

import mediapipe as mp

# ---------- Utils ----------
def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def pixelate_region(img, x, y, w, h, blocks=16):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return img
    H, W = roi.shape[:2]
    bh = max(1, H // blocks)
    bw = max(1, W // blocks)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

def blur_region(img, x, y, w, h, k=25):
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return img
    img[y:y+h, x:x+w] = cv2.blur(roi, (k, k))
    return img

def overlay_emoji(img, x, y, w, h, emoji_path):
    if not emoji_path or not Path(emoji_path).exists():
        return img
    emoji = cv2.imread(str(emoji_path), cv2.IMREAD_UNCHANGED)
    if emoji is None:
        return img
    # Resize emoji to bbox
    em = cv2.resize(emoji, (w, h), interpolation=cv2.INTER_AREA)
    # If has alpha channel, blend
    if em.shape[2] == 4:
        alpha = em[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+h, x:x+w, c] = (alpha * em[:, :, c] + (1 - alpha) * img[y:y+h, x:x+w, c]).astype(img.dtype)
    else:
        img[y:y+h, x:x+w] = em
    return img

# ---------- Detector ----------
class MPFaceDetector:
    def __init__(self, model_selection=1, min_conf=0.5):
        self.model_selection = model_selection
        self.min_conf = min_conf
        self._ctx = mp.solutions.face_detection

    def __enter__(self):
        self.det = self._ctx.FaceDetection(model_selection=self.model_selection,
                                           min_detection_confidence=self.min_conf)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.det.close()

    def detect(self, frame):
        # MediaPipe espera RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.det.process(rgb)
        boxes = []
        if result.detections:
            H, W = frame.shape[:2]
            for d in result.detections:
                bbox = d.location_data.relative_bounding_box
                x1 = int(bbox.xmin * W)
                y1 = int(bbox.ymin * H)
                w  = int(bbox.width * W)
                h  = int(bbox.height * H)
                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                w = max(0, min(W - x1, w)); h = max(0, min(H - y1, h))
                boxes.append((x1, y1, w, h))
        return boxes

# ---------- Core ----------
def anonymize(frame, boxes, method="blur", draw=False, emoji_path=None):
    for (x, y, w, h) in boxes:
        if method == "pixelate":
            frame = pixelate_region(frame, x, y, w, h, blocks=16)
        elif method == "emoji":
            frame = overlay_emoji(frame, x, y, w, h, emoji_path)
        else:
            frame = blur_region(frame, x, y, w, h, k=25)
        if draw:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def load_config(cfg_path: Path | None):
    defaults = {"detector": {"model_selection": 1, "min_conf": 0.5}}
    if not cfg_path or not cfg_path.exists():
        return defaults
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # shallow-merge
    defaults["detector"].update(user_cfg.get("detector", {}))
    return defaults

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Face anonymizer (image/video) with blur/pixelate/emoji modes.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=str, help="Path to an input image or video.")
    src.add_argument("--webcam", "-w", type=int, nargs="?", const=0, help="Webcam index (default 0).")
    p.add_argument("--method", "-m", choices=["blur", "pixelate", "emoji"], default="blur", help="Anonymization method.")
    p.add_argument("--emoji", type=str, default=None, help="Path to an emoji PNG (optional, for --method emoji).")
    p.add_argument("--draw", action="store_true", help="Draw face bounding boxes.")
    p.add_argument("--output", "-o", type=str, help="Output file path (image or video). If omitted, preview only.")
    p.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="YAML config for detector.")
    return p.parse_args()

def is_video_path(p: Path):
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def run():
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    det_cfg = cfg["detector"]

    with MPFaceDetector(det_cfg["model_selection"], det_cfg["min_conf"]) as fd:
        if args.webcam is not None:
            cap = cv2.VideoCapture(args.webcam)
            writer = None
            if args.output:
                ensure_dir(Path(args.output))
                # Fallback a 640x480 si no está listo el stream aún
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                boxes = fd.detect(frame)
                frame = anonymize(frame, boxes, method=args.method, draw=args.draw, emoji_path=args.emoji)
                if writer:
                    writer.write(frame)
                cv2.imshow("Face Anonymizer (press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
        else:
            inp = Path(args.input)
            if not inp.exists():
                raise FileNotFoundError(f"Input not found: {inp}")
            if is_video_path(inp):
                cap = cv2.VideoCapture(str(inp))
                writer = None
                if args.output:
                    ensure_dir(Path(args.output))
                    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    boxes = fd.detect(frame)
                    frame = anonymize(frame, boxes, method=args.method, draw=args.draw, emoji_path=args.emoji)
                    if writer:
                        writer.write(frame)
                    cv2.imshow("Face Anonymizer (press Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cap.release()
                if writer: writer.release()
                cv2.destroyAllWindows()
            else:
                frame = cv2.imread(str(inp))
                if frame is None:
                    raise ValueError(f"Could not read image: {inp}")
                boxes = fd.detect(frame)
                frame = anonymize(frame, boxes, method=args.method, draw=args.draw, emoji_path=args.emoji)
                if args.output:
                    outp = Path(args.output)
                    ensure_dir(outp)
                    cv2.imwrite(str(outp), frame)
                cv2.imshow("Face Anonymizer (press any key to close)", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
