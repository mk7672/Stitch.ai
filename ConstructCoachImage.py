"""
Last updated: 05 Mar 26
Created by  : Prof Arbind Kumar Gupta
Description : Fixed-trackside-camera stitcher.

              Folder structure:
                <root>/<train_number>/<coach_number>/  ← images of one coach passing
              Each coach folder produces exactly ONE stitched panorama.

              Algorithm:
                1. Compute phase-correlation dx for every consecutive frame pair
                2. Estimate true per-frame pixel shift from reliable measurements
                3. Correct raw dx for dropped/skipped frames (n = round(dx/true_dx))
                4. Find the FIRST and LAST frame pair where the train is present
                   (|dx| > DX_MIN). Everything in between — including momentary
                   phase-correlation failures — is filled with the local mean step.
                5. Strip-scan (push-broom) stitch: each frame contributes a
                   right-edge slice of width = corrected_step. No overlap → no ghosting.
                6. Save ONE PNG + ONE speed CSV per coach folder.

Version     : 14.0
"""

import os, csv
import cv2
import numpy as np
from pathlib import Path

# ======================== CONFIGURATION ========================
# Set up input/output directories, cropping, and algorithm parameters
_HERE           = Path(__file__).parent
BASE_INPUT_DIR  = _HERE / 'feb_11'  # Root directory containing coach folders
BASE_OUTPUT_DIR = _HERE / '09-03-2026 Outputs - Basic Stitching - HPDS'   # Output directory for stitched images and CSVs
COACH_DEPTH     = 1          # Folder depth for coach folders (1 or 2)

TOP_CROP        = 00         # Number of rows to crop from top of each frame
CLIP_FRAC       = 0.03       # Fraction of image width to clip from left/right before correlation

DX_MIN          = 30         # Minimum pixel shift to consider train present
DX_MAX_SINGLE   = 220        # Maximum plausible single-frame shift
DX_REJECT       = 800        # Discard measurements above this threshold
RESP_MIN        = 0.03       # Minimum phase-correlation response to trust

# --- Fisheye Correction Parameters ---
# Set these to your camera's calibration values
APPLY_FISHEYE_CORRECTION = True
CAMERA_MATRIX = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)  # Example values
DIST_COEFFS = np.array([-0.1, 0.01, 0, 0], dtype=np.float32)  # Example values
# ===============================================================

# ------------------- Fisheye Correction ------------------------
def undistort_fisheye(img: np.ndarray) -> np.ndarray:
    """
    Correct fisheye distortion in the input image using the camera matrix and distortion coefficients.
    Adjust CAMERA_MATRIX and DIST_COEFFS above to match your camera calibration.
    """
    h, w = img.shape[:2]
    new_camera_mtx = CAMERA_MATRIX.copy()
    # Optionally, optimize new_camera_mtx for undistortion (can use cv2.getOptimalNewCameraMatrix)
    undistorted = cv2.undistort(img, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    return undistorted

# ------------------- Image Preprocessing ----------------------
def cropTop(im: np.ndarray, rows: int = TOP_CROP) -> np.ndarray:
    """Crop the top rows from an image."""
    return im[rows:, :] if rows > 0 else im


def clipImg(im: np.ndarray, frac: float = CLIP_FRAC) -> np.ndarray:
    """Clip a fraction of the image width from both left and right sides."""
    if frac <= 0:
        return im
    c = max(1, int(im.shape[1] * frac))
    return im[:, c:-c] if 2 * c < im.shape[1] else im

# ------------------- Image Loading ----------------------------
def getImages(fldr: Path) -> tuple[list, list]:
    """Load all valid images from a folder, apply fisheye correction, crop top rows, and return images and filenames."""
    imgs, names = [], []
    for f in sorted(os.listdir(fldr)):
        if not any(f.lower().endswith(e) for e in ['.jpg', '.jpeg', '.png', '.bmp']):
            continue
        img = cv2.imread(str(fldr / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if APPLY_FISHEYE_CORRECTION:
            img = undistort_fisheye(img)
        imgs.append(cropTop(img))
        names.append(f)
    return imgs, names

# ------------------- Phase Correlation ------------------------
def phaseShift(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """Compute phase-correlation shift and response between two images."""
    a = clipImg(img1).astype(np.float32)
    b = clipImg(img2).astype(np.float32)
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, _), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(resp)

# ------------------- True Shift Estimation --------------------
def estimateTrueDx(raw_dxs: list, responses: list) -> float:
    """Estimate the true per-frame pixel shift using reliable measurements."""
    clean = [abs(d) for d, r in zip(raw_dxs, responses)
             if DX_MIN <= abs(d) <= DX_MAX_SINGLE and r >= RESP_MIN]
    if clean:
        return float(np.median(clean))
    above = [abs(d) for d in raw_dxs if DX_MIN < abs(d) < DX_REJECT]
    return float(np.min(above)) if above else 143.0

# ------------------- Step Correction --------------------------
def rawToStep(raw_dx: float, resp: float, true_dx: float) -> tuple[float, int]:
    """Convert raw dx to corrected per-frame step and number of frames skipped."""
    adx = abs(raw_dx)
    if adx < DX_MIN or adx > DX_REJECT or resp < RESP_MIN:
        return 0.0, 0
    n = max(1, round(adx / true_dx))
    return adx / n, n

# ------------------- Main Processing Function -----------------
def processCoachFolder(input_folder: Path, output_folder: Path) -> None:
    """
    Process a single coach folder:
    - Loads images
    - Computes phase-correlation shifts
    - Estimates true shift
    - Corrects for dropped/skipped frames
    - Identifies train-present frames
    - Fills gaps in measurements
    - Stitches images using strip-scan method
    - Saves outputs (PNG and CSV)
    """
    print(f'\n{"="*72}')
    print(f'  Coach folder : {input_folder}')

    # Load and preprocess images
    imgs, names = getImages(input_folder)
    n_imgs = len(imgs)
    if n_imgs < 2:
        print(f'  [SKIP] Only {n_imgs} image(s).')
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f'  {n_imgs} frames loaded.')

    # 1. Phase correlation for every consecutive pair
    raw_dxs, raw_resps = [], []
    for i in range(n_imgs - 1):
        dx, resp = phaseShift(imgs[i], imgs[i + 1])
        raw_dxs.append(dx)
        raw_resps.append(resp)

    # 2. Estimate true per-frame shift
    true_dx = estimateTrueDx(raw_dxs, raw_resps)
    print(f'  True dx/frame : {true_dx:.1f} px')

    # 3. Correct each pair for dropped/skipped frames
    steps_raw, n_frames_list = [], []
    for dx, resp in zip(raw_dxs, raw_resps):
        s, n = rawToStep(dx, resp, true_dx)
        steps_raw.append(s)
        n_frames_list.append(n)

    # 4. Find first and last active pair (train present)
    active_pairs = [i for i, s in enumerate(steps_raw) if s > 0]
    if not active_pairs:
        print('  [SKIP] No train-present frames detected.')
        return

    first_pair = active_pairs[0]
    last_pair  = active_pairs[-1]
    print(f'  Train visible: pairs {first_pair} → {last_pair}  '
          f'({last_pair - first_pair + 2} frames)')

    # 5. Fill gaps in step measurements with local mean
    filled_steps = list(steps_raw)
    for i in range(first_pair, last_pair + 1):
        if filled_steps[i] == 0:
            # Fill with mean of nearest nonzero neighbours
            left  = [filled_steps[j] for j in range(first_pair, i) if filled_steps[j] > 0]
            right = [filled_steps[j] for j in range(i + 1, last_pair + 1) if filled_steps[j] > 0]
            neighbours = (left[-3:] if left else []) + (right[:3] if right else [])
            filled_steps[i] = float(np.mean(neighbours)) if neighbours else true_dx

    mean_step = float(np.mean([filled_steps[i] for i in range(first_pair, last_pair + 1)]))
    print(f'  Mean step     : {mean_step:.1f} px')

    # 6. Save full pair CSV with all measurements
    csv_all = output_folder / f'{input_folder.name}_all_pairs.csv'
    with open(csv_all, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pair', 'frame_a', 'frame_b', 'raw_dx', 'response',
                    'true_dx_est', 'n_frames', 'corrected_step'])
        for i, (dx, resp, s, n) in enumerate(
                zip(raw_dxs, raw_resps, steps_raw, n_frames_list)):
            w.writerow([i, i, i + 1, f'{dx:.2f}', f'{resp:.4f}',
                        f'{true_dx:.1f}', n, f'{s:.2f}'])
    print(f'  Saved all-pairs CSV → {csv_all}')

    # 7. Strip-scan stitch: build panorama by concatenating right-edge strips
    step_ints = [max(1, int(round(filled_steps[p])))
                 for p in range(first_pair, last_pair + 1)]

    frame_start = first_pair        # index into imgs[]
    frame_end   = last_pair + 1     # inclusive

    first_clip = clipImg(imgs[frame_start])
    h0, img_w  = first_clip.shape

    total_w = img_w + sum(step_ints)
    canvas  = np.zeros((h0, total_w), dtype=np.uint8)

    # Place the first frame (clipped) at the start of the canvas
    canvas[:h0, :img_w] = first_clip[:h0, :]
    x_cursor = img_w

    # For each subsequent frame, add a right-edge strip of width = corrected step
    for i, fi in enumerate(range(frame_start + 1, frame_end + 1)):
        if fi >= n_imgs:
            break
        img = clipImg(imgs[fi])
        h_i = min(img.shape[0], h0)
        sw  = min(step_ints[i], img.shape[1])
        strip = img[:h_i, img.shape[1] - sw:]
        sw_actual = strip.shape[1]
        canvas[:h_i, x_cursor:x_cursor + sw_actual] = strip
        x_cursor += sw_actual

    stitched = canvas[:, :x_cursor]

    # 8. Save outputs: stitched PNG and speed CSV
    out_png = output_folder / f'{input_folder.name}.png'
    out_csv = output_folder / f'{input_folder.name}_speed.csv'

    cv2.imwrite(str(out_png), stitched)
    print(f'  Stitched PNG  → {out_png}  ({stitched.shape[1]} × {stitched.shape[0]} px)')

    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['transition', 'frame_a', 'frame_b',
                     'raw_dx', 'n_frames', 'corrected_step', 'gap_filled'])
        for j, p in enumerate(range(first_pair, last_pair + 1)):
            gap_filled = int(steps_raw[p] == 0 and filled_steps[p] > 0)
            wr.writerow([j, p, p + 1,
                         f'{raw_dxs[p]:.2f}',
                         n_frames_list[p],
                         f'{filled_steps[p]:.2f}',
                         gap_filled])
    print(f'  Speed CSV     → {out_csv}')

# ------------------- Coach Folder Discovery -------------------
def find_coach_folders(base: Path, depth: int) -> list:
    """Recursively find coach folders at the specified depth."""
    if depth == 1:
        return [p for p in sorted(base.iterdir()) if p.is_dir()]
    leaves = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            leaves.extend(find_coach_folders(child, depth - 1))
    return leaves

# ------------------- Main Entry Point -------------------------
def main():
    """
    Main function:
    - Checks input directory
    - Finds coach folders
    - Processes each coach folder
    - Prints summary when done
    """
    if not BASE_INPUT_DIR.exists():
        print(f'[ERROR] BASE_INPUT_DIR not found: {BASE_INPUT_DIR.resolve()}')
        return

    coach_folders = find_coach_folders(BASE_INPUT_DIR, COACH_DEPTH)
    print(f'Found {len(coach_folders)} coach folder(s) under {BASE_INPUT_DIR}')
    print(f'Config: DX_MIN={DX_MIN}  DX_MAX_SINGLE={DX_MAX_SINGLE}  '
          f'DX_REJECT={DX_REJECT}  RESP_MIN={RESP_MIN}\n')

    for coach_dir in coach_folders:
        # Compute output path and process each coach folder
        rel_path   = coach_dir.relative_to(BASE_INPUT_DIR)
        out_parent = BASE_OUTPUT_DIR / rel_path.parent
        processCoachFolder(coach_dir, out_parent)

    print(f'\n{"="*72}')
    print('All done.')

# ------------------- Script Execution -------------------------
if __name__ == '__main__':
    main()