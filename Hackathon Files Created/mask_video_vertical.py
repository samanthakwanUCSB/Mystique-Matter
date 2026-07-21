#!/usr/bin/env python3
"""mask_video_vertical.py

Enhanced video vertical masking tool.

Usage examples are in the README on your Desktop. The script keeps a vertical
band of the frame and blacks (or fades) the sides. Supports hard and soft
masks (gradient/alpha).
"""

import argparse
import os
import sys
from typing import Tuple

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Mask a video vertically and save to Desktop")
    p.add_argument("input", nargs='?', help="Path to input video file (optional). If omitted, you'll be prompted")
    p.add_argument("--fraction", type=float, default=0.5,
                   help="Fraction of the width to keep (0-1). Default 0.5")
    p.add_argument("--position", choices=("center", "left", "right"), default="center",
                   help="Position of the kept vertical band. Default: center")
    p.add_argument("--mode", choices=("hard", "soft"), default="hard",
                   help="Mask style: 'hard' zeros outside band, 'soft' uses a horizontal gradient transition")
    p.add_argument("--transition", type=float, default=0.05,
                   help="Transition width as fraction of frame width for soft mode (0-0.5). Default 0.05")
    p.add_argument("--output", help="Optional output path. If omitted, saved to Desktop with _masked.mp4 suffix")
    return p.parse_args()


def build_output_path(input_path: str, out_arg: str | None) -> str:
    if out_arg:
        return out_arg
    base = os.path.splitext(os.path.basename(input_path))[0]
    desktop = os.path.expanduser("~/Desktop")
    return os.path.join(desktop, f"{base}_masked.mp4")


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def make_alpha_mask(width: int, height: int, keep_x0: int, keep_x1: int,
                    mode: str = "hard", transition_frac: float = 0.05) -> np.ndarray:
    """Return an (H, W) float32 alpha mask in range [0,1]. 1==keep, 0==black."""
    mask = np.zeros((height, width), dtype=np.float32)
    # central full area
    mask[:, keep_x0:keep_x1] = 1.0

    if mode == "hard":
        return mask

    # soft mode: add linear ramps on both sides
    trans = clamp(transition_frac, 0.0, 0.5) * width
    trans = int(round(trans))
    if trans <= 0:
        return mask

    # left ramp
    left_start = max(0, keep_x0 - trans)
    if left_start < keep_x0:
        ramp = np.linspace(0.0, 1.0, keep_x0 - left_start, endpoint=False, dtype=np.float32)
        mask[:, left_start:keep_x0] = ramp[np.newaxis, :]

    # right ramp
    right_end = min(width, keep_x1 + trans)
    if keep_x1 < right_end:
        ramp = np.linspace(1.0, 0.0, right_end - keep_x1, endpoint=False, dtype=np.float32)
        mask[:, keep_x1:right_end] = ramp[np.newaxis, :]

    return mask


def process_video(inp: str, out_path: str, fraction: float, position: str,
                  mode: str, transition: float) -> None:
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keep_w = int(round(width * fraction))
    if keep_w <= 0:
        raise ValueError("Mask fraction too small, no width to keep.")

    if position == 'center':
        x0 = (width - keep_w) // 2
    elif position == 'left':
        x0 = 0
    else:  # right
        x0 = width - keep_w
    x1 = x0 + keep_w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output for writing: {out_path}")

    print(f"Processing: {inp}")
    print(f"Output: {out_path}")
    print(f"Video size: {width}x{height} @ {fps}fps | Keeping columns {x0}:{x1} | mode={mode}")

    # precompute alpha mask
    alpha = make_alpha_mask(width, height, x0, x1, mode=mode, transition_frac=transition)
    # convert to 3-channel for multiplication
    alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Frame is BGR uint8. Convert to float in 0..1
            f = frame.astype(np.float32) / 255.0
            # black background is zero; apply alpha (kept region stays, sides fade to black)
            out_frame = (f * alpha_3)
            # convert back to uint8
            out_bgr = (out_frame * 255.0).clip(0, 255).astype(np.uint8)
            out.write(out_bgr)
            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"Processed {frame_idx} frames...")
    finally:
        cap.release()
        out.release()

    print(f"Done — processed {frame_idx} frames.")


def main():
    args = parse_args()
    inp = args.input
    # If no input provided via CLI, prompt the user (useful when running in VS Code)
    if not inp:
        try:
            inp = input("Enter full path to your video file (e.g. /Users/you/Desktop/input.mp4): ").strip()
        except EOFError:
            print("No input provided.")
            sys.exit(2)

    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}")
        sys.exit(2)

    fraction = clamp(args.fraction, 0.0, 1.0)
    transition = clamp(args.transition, 0.0, 0.5)
    out_path = build_output_path(inp, args.output)

    try:
        process_video(inp, out_path, fraction, args.position, args.mode, transition)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
