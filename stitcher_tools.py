"""Local, cancellable helpers for the Streamlit coach stitcher.

This module mirrors the algorithm in ConstructCoachImage.py without mutating
that script's module-level settings.  It also creates small synthetic coach
folders that can be selected as normal stitcher input.
"""

from __future__ import annotations

import csv
import math
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class Cancelled(Exception):
    """Raised at a safe cancellation checkpoint."""


def discover_coaches(root: Path, depth: int) -> list[Path]:
    if depth not in (1, 2) or not root.is_dir():
        return []
    if depth == 1:
        return sorted((p for p in root.iterdir() if p.is_dir()))
    return sorted(
        (leaf for parent in root.iterdir() if parent.is_dir()
         for leaf in parent.iterdir() if leaf.is_dir()),
        key=lambda p: (p.parent.name, p.name),
    )


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled by user")


def _clip(img: np.ndarray, frac: float) -> np.ndarray:
    if frac <= 0:
        return img
    c = max(1, int(img.shape[1] * frac))
    return img[:, c:-c] if 2 * c < img.shape[1] else img


def _load_images(folder: Path, config: dict, cancel: threading.Event | None = None):
    imgs: list[np.ndarray] = []
    names: list[str] = []
    # Match ConstructCoachImage.getImages(): sort the names with Python's
    # normal case-sensitive ordering before loading supported image files.
    for file in sorted(folder.iterdir()):
        _check_cancel(cancel)
        if file.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if config["fisheye"]:
            img = cv2.undistort(img, config["camera_matrix"], config["dist_coeffs"],
                                None, config["camera_matrix"])
        crop = config["top_crop"]
        if crop > 0:
            img = img[crop:, :]
        imgs.append(img)
        names.append(file.name)
    return imgs, names


def _phase_shift(img1: np.ndarray, img2: np.ndarray, clip_frac: float) -> tuple[float, float]:
    a = _clip(img1, clip_frac).astype(np.float32)
    b = _clip(img2, clip_frac).astype(np.float32)
    h, w = a.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, _), response = cv2.phaseCorrelate(a, b, window)
    return float(dx), float(response)


def analyze_coach(folder: Path, config: dict,
                  cancel: threading.Event | None = None,
                  include_images: bool = False) -> dict:
    """Load and analyze pairs only. No panorama or output files are created."""
    imgs, names = _load_images(folder, config, cancel)
    if len(imgs) < 2:
        return {"status": "skipped", "reason": f"Only {len(imgs)} readable image(s).",
                "records": [], "frame_count": len(imgs), "preview_frame": imgs[0] if imgs else None}

    raw_dxs: list[float] = []
    responses: list[float] = []
    for index in range(len(imgs) - 1):
        _check_cancel(cancel)
        dx, response = _phase_shift(imgs[index], imgs[index + 1], config["clip_frac"])
        raw_dxs.append(dx)
        responses.append(response)

    clean = [abs(dx) for dx, response in zip(raw_dxs, responses)
             if config["dx_min"] <= abs(dx) <= config["dx_max"]
             and response >= config["resp_min"]]
    if clean:
        true_dx = float(np.median(clean))
    else:
        above = [abs(dx) for dx in raw_dxs
                 if config["dx_min"] < abs(dx) < config["dx_reject"]]
        true_dx = float(min(above)) if above else 143.0

    raw_steps: list[float] = []
    n_frames: list[int] = []
    for dx, response in zip(raw_dxs, responses):
        adx = abs(dx)
        if adx < config["dx_min"] or adx > config["dx_reject"] or response < config["resp_min"]:
            raw_steps.append(0.0)
            n_frames.append(0)
        else:
            n = max(1, round(adx / true_dx))
            raw_steps.append(adx / n)
            n_frames.append(n)

    active = [i for i, step in enumerate(raw_steps) if step > 0]
    if active:
        first, last = active[0], active[-1]
    else:
        first, last = None, None
    filled = list(raw_steps)
    if first is not None:
        # Match ConstructCoachImage.processCoachFolder(): every zero step
        # between the first and last active pair is filled from up to three
        # nearest nonzero steps on either side, with no run-length limit.
        for pair in range(first, last + 1):
            if filled[pair] == 0:
                left = [filled[j] for j in range(first, pair) if filled[j] > 0]
                right = [filled[j] for j in range(pair + 1, last + 1) if filled[j] > 0]
                neighbours = (left[-3:] if left else []) + (right[:3] if right else [])
                filled[pair] = float(np.mean(neighbours)) if neighbours else true_dx

    records = []
    for i, (dx, response, step, n) in enumerate(zip(raw_dxs, responses, raw_steps, n_frames)):
        inside = first is not None and first <= i <= last
        if step > 0:
            category = "train-present"
        elif inside and filled[i] > 0:
            category = "gap-filled"
        else:
            category = "rejected"
        records.append({
            "pair": i, "frame_a": names[i], "frame_b": names[i + 1],
            "raw_dx": dx, "response": response, "true_dx_est": true_dx,
            "n_frames": n, "corrected_step": step,
            "filled_step": filled[i], "category": category,
        })
    status = "ok" if active else "skipped"
    reason = "" if active else "No train-present frame pairs detected."
    result = {
        "status": status, "reason": reason, "records": records,
        "frame_count": len(imgs), "true_dx": true_dx,
        "first_pair": first, "last_pair": last, "filled_steps": filled,
        "raw_steps": raw_steps, "n_frames": n_frames,
        "preview_frame": imgs[len(imgs) // 2],
    }
    if include_images:
        result["_images"] = imgs
    return result


def build_panorama(imgs: list[np.ndarray], analysis: dict, config: dict,
                   cancel: threading.Event | None = None) -> np.ndarray:
    """Run the strip-scan stitch in memory, without writing output files."""
    first, last = analysis["first_pair"], analysis["last_pair"]
    if first is None or last is None:
        raise ValueError("No train-present pairs are available to stitch.")
    filled = analysis["filled_steps"]
    step_ints = [max(1, int(round(filled[p]))) for p in range(first, last + 1) if filled[p] > 0]
    first_clip = _clip(imgs[first], config["clip_frac"])
    height0, image_width = first_clip.shape
    canvas = np.zeros((height0, image_width + sum(step_ints)), dtype=np.uint8)
    canvas[:, :image_width] = first_clip
    cursor = image_width
    for pair in range(first, last + 1):
        _check_cancel(cancel)
        step = filled[pair]
        if step <= 0:
            continue
        frame_index = pair + 1
        if frame_index >= len(imgs):
            break
        img = _clip(imgs[frame_index], config["clip_frac"])
        height = min(img.shape[0], height0)
        strip_width = min(max(1, int(round(step))), img.shape[1])
        strip = img[:height, img.shape[1] - strip_width:]
        actual_width = strip.shape[1]
        canvas[:height, cursor:cursor + actual_width] = strip
        cursor += actual_width
    _check_cancel(cancel)
    return canvas[:, :cursor]


def process_coach(folder: Path, output_folder: Path, config: dict,
                  cancel: threading.Event | None = None,
                  log: Callable[[str], None] | None = None) -> dict:
    def say(message: str) -> None:
        if log:
            log(message)

    say(f"Reading {folder}")
    analysis = analyze_coach(folder, config, cancel, include_images=True)
    if analysis["status"] != "ok":
        say(f"Skipped {folder.name}: {analysis['reason']}")
        return {"status": "skipped", "reason": analysis["reason"], "folder": str(folder)}

    imgs = analysis.pop("_images")
    first, last = analysis["first_pair"], analysis["last_pair"]
    filled = analysis["filled_steps"]
    stitched = build_panorama(imgs, analysis, config, cancel)

    output_folder.mkdir(parents=True, exist_ok=True)
    png_path = output_folder / f"{folder.name}.png"
    all_csv = output_folder / f"{folder.name}_all_pairs.csv"
    speed_csv = output_folder / f"{folder.name}_speed.csv"
    records = analysis["records"]
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "frame_a", "frame_b", "raw_dx", "response",
                         "true_dx_est", "n_frames", "corrected_step"])
        for row in records:
            writer.writerow([row["pair"], row["pair"], row["pair"] + 1,
                             f"{row['raw_dx']:.2f}", f"{row['response']:.4f}",
                             f"{row['true_dx_est']:.1f}", row["n_frames"],
                             f"{row['corrected_step']:.2f}"])
    if not cv2.imwrite(str(png_path), stitched):
        raise OSError(f"Could not write panorama: {png_path}")
    with speed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["transition", "frame_a", "frame_b", "raw_dx",
                         "n_frames", "corrected_step", "gap_filled"])
        for transition, pair in enumerate(range(first, last + 1)):
            row = records[pair]
            writer.writerow([transition, pair, pair + 1, f"{row['raw_dx']:.2f}",
                             row["n_frames"], f"{filled[pair]:.2f}",
                             int(analysis["raw_steps"][pair] == 0 and filled[pair] > 0)])
    say(f"Saved panorama and CSVs for {folder.name}")
    return {"status": "ok", "reason": "", "folder": str(folder), "png": str(png_path),
            "all_pairs_csv": str(all_csv), "speed_csv": str(speed_csv),
            "frame_count": analysis["frame_count"]}


def generate_synthetic_dataset(root: Path, depth: int, coach_count: int,
                               frame_count: int, width: int, height: int,
                               step: int, gap_every: int = 0, seed: int = 7) -> list[Path]:
    """Create grayscale moving-train frames that can be read by the stitcher."""
    if depth not in (1, 2):
        raise ValueError("Depth must be 1 or 2.")
    if coach_count < 1 or frame_count < 3 or width < 320 or height < 160 or step < 1:
        raise ValueError("Use at least 1 coach, 3 frames, 320×160 pixels and a positive step.")
    rng = np.random.default_rng(seed)
    made: list[Path] = []
    root.mkdir(parents=True, exist_ok=True)
    for coach_index in range(coach_count):
        coach_name = f"coach_{coach_index + 1:03d}"
        coach_dir = root / coach_name if depth == 1 else root / f"train_{coach_index + 1:03d}" / coach_name
        coach_dir.mkdir(parents=True, exist_ok=True)

        transitions = [step] * (frame_count - 1)
        if gap_every > 0 and frame_count > 4:
            for index in range(gap_every - 1, frame_count - 2, gap_every):
                transitions[index] = 0
                transitions[index + 1] = 2 * step
        positions = [0]
        for amount in transitions:
            positions.append(positions[-1] + amount)
        scene_width = width + positions[-1] + 4
        scene = np.full((height, scene_width), 42, dtype=np.uint8)
        body_top, body_bottom = int(height * .22), int(height * .78)
        body_color = int(118 + coach_index * 7 % 45)
        cv2.rectangle(scene, (0, body_top), (scene_width - 1, body_bottom), body_color, -1)
        cv2.line(scene, (0, body_top), (scene_width - 1, body_top), 220, max(2, height // 150))
        cv2.line(scene, (0, body_bottom), (scene_width - 1, body_bottom), 25, max(2, height // 100))
        window_w = max(18, width // 14)
        window_h = max(18, (body_bottom - body_top) // 3)
        pitch = max(window_w + 24, width // 7)
        for x in range(18, scene_width, pitch):
            cv2.rectangle(scene, (x, body_top + 35),
                          (min(x + window_w, scene_width - 1), body_top + 35 + window_h),
                          int(24 + (x // max(1, pitch) * 17) % 40), -1)
            cv2.rectangle(scene, (x, body_top + 35),
                          (min(x + window_w, scene_width - 1), body_top + 35 + window_h), 205, 2)
            cv2.line(scene, (x + window_w // 2, body_top + 35),
                     (x + window_w // 2, min(body_top + 35 + window_h, height - 1)), 160, 1)
        for x in range(0, scene_width, max(80, width // 5)):
            cv2.line(scene, (x, body_top + 5), (x + 12, body_bottom - 5), 75, 2)

        for frame_index, x in enumerate(positions):
            frame = scene[:, x:x + width].copy()
            if frame.shape[1] < width:
                frame = cv2.copyMakeBorder(frame, 0, 0, 0, width - frame.shape[1], cv2.BORDER_REPLICATE)
            noise = rng.normal(0, 1.1, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            path = coach_dir / f"frame_{frame_index:05d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise OSError(f"Could not write synthetic frame: {path}")
        made.append(coach_dir)
    return made
