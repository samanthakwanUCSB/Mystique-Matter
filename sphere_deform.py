#!/usr/bin/env python3
"""
============================================================
 Sphere Deformation Through a Channel
============================================================
Detect circular spheres (dark outlines, no fill contrast) moving vertically
through a channel, STITCH fragmented tracks into true per-sphere trajectories,
keep only the real spheres, label them 1..N by the order they ENTER from the
top, and report:

  * deformation in the X direction (width, across the channel)
  * deformation in the Y direction (length, along the motion)
    where  deformation% = (current - baseline) / baseline * 100   (signed)
      - negative X  = compressed across the channel
      - positive Y  = elongated along the motion

Outputs (saved to Desktop):
  * <name>_deformation.csv   -> per-frame X/Y deformation for every sphere
  * <name>_dimensions.csv    -> per-sphere dimensional summary (studied sizes)
  * <name>_deformation.png   -> combined graph, one colour per sphere
  * <name>_per_sphere.png    -> one small panel per sphere
  * <name>_annotated.mp4      -> video with tracking boxes to verify detection

Run:
    python3 sphere_deform.py                 (asks for the path)
    python3 sphere_deform.py /path/file.tif  (path as argument)
============================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import distance as dist

# ============================================================
#  SETTINGS
# ============================================================
DP              = 1.2
PARAM1          = 100
PARAM2          = 30
MATCH_DIST      = 80
MAX_MISSING     = 25
BASELINE_N      = 3
DARK_PERCENTILE = 30
CHANNEL_TOP_Y   = None      # channel entrance y-px; None = auto (40% of height)

# ---- Stitching / filtering (repairs fragmented tracks) ----
STITCH_MAX_TIME_GAP  = 40   # frames: max gap between a fragment's end and next's start
STITCH_MAX_DIST      = 120  # px: max centroid jump across that gap
STITCH_MAX_X_DRIFT   = 40   # px: spheres move mostly vertically; limit x wandering
MIN_TRANSIT_FRAMES   = 8    # a real sphere is seen at least this many frames
MIN_VERTICAL_TRAVEL  = None # px a real sphere must travel down; None = auto (25% H)

# ---- Optional calibration ----
UM_PER_PX = None            # set e.g. 0.5 to report dimensions in micrometers
# ============================================================


# ------------------------------------------------------------
#  INPUT
# ------------------------------------------------------------
def get_input_path():
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter the video/TIFF path: ")
    path = os.path.expanduser(path.strip().strip('"').strip("'"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")
    return path


def load_frames(path):
    """Return (list_of_grayscale_uint8_frames, fps). Handles video or TIFF stack."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        stack = np.asarray(tifffile.imread(path))
        if stack.ndim == 2:
            stack = stack[None, ...]
        elif stack.ndim == 3 and stack.shape[-1] in (3, 4):
            stack = stack[None, ...]
        frames = []
        for fr in stack:
            if fr.ndim == 3:
                fr = cv2.cvtColor(fr[..., :3], cv2.COLOR_RGB2GRAY)
            if fr.dtype != np.uint8:
                fr = fr.astype(np.float64)
                mn, mx = fr.min(), fr.max()
                fr = (fr - mn) / (mx - mn) * 255 if mx > mn else np.zeros_like(fr)
                fr = fr.astype(np.uint8)
            frames.append(np.ascontiguousarray(fr))
        return frames, 20.0     # TIFF stacks carry no fps; edit if you know it
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, fps


# ------------------------------------------------------------
#  DETECTION HELPERS
# ------------------------------------------------------------
def estimate_radius_range(frames):
    radii = []
    for g in frames[:min(len(frames), 15)]:
        gb = cv2.medianBlur(g, 5)
        h = gb.shape[0]
        circles = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, dp=DP, minDist=h // 20,
                                   param1=PARAM1, param2=PARAM2,
                                   minRadius=5, maxRadius=h // 4)
        if circles is not None:
            radii.extend(circles[0, :, 2].tolist())
    if not radii:
        h = frames[0].shape[0]
        return max(5, h // 40), h // 6
    r_med = np.median(radii)
    r_min, r_max = max(4, int(r_med * 0.5)), int(r_med * 1.8)
    print(f"Auto radius: median ~{r_med:.1f}px (search {r_min}-{r_max}px)")
    return r_min, r_max


def measure_boundary(gray, cx, cy, r_search, dark_thresh):
    """Scan outward from center to the dark boundary ring in x and y.
    Returns (x_dimension, y_dimension) = full extents in pixels."""
    H, W = gray.shape
    cx, cy = int(round(cx)), int(round(cy))
    maxd = int(r_search * 1.8)
    def scan(dx, dy):
        for d in range(3, maxd):
            x, y = cx + dx * d, cy + dy * d
            if x < 0 or y < 0 or x >= W or y >= H:
                return d
            if gray[y, x] <= dark_thresh:
                return d
        return maxd
    left, right = scan(-1, 0), scan(1, 0)
    up, down = scan(0, -1), scan(0, 1)
    return float(left + right), float(up + down)


# ------------------------------------------------------------
#  TRACKING
# ------------------------------------------------------------
class Track:
    _next = 0
    def __init__(self, cx, cy, fi, w, l):
        self.id = Track._next; Track._next += 1
        self.cx, self.cy = cx, cy
        self.vx, self.vy = 0.0, 0.0
        self.records = [(fi, cx, cy, w, l)]
        self.missing = 0
        self.label = None
    def predict(self):
        return self.cx + self.vx, self.cy + self.vy
    def update(self, cx, cy, w, l, fi):
        self.vx = 0.6 * self.vx + 0.4 * (cx - self.cx)
        self.vy = 0.6 * self.vy + 0.4 * (cy - self.cy)
        self.cx, self.cy = cx, cy
        self.records.append((fi, cx, cy, w, l))
        self.missing = 0
    def coast(self):
        self.cx, self.cy = self.predict()
        self.missing += 1


def run_tracking(frames, r_min, r_max, dark_thresh, channel_top):
    all_tracks, active, annotated = [], [], []
    H, W = frames[0].shape
    for fi, gray in enumerate(frames):
        gb = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT, dp=DP,
                                   minDist=int(r_min * 1.5),
                                   param1=PARAM1, param2=PARAM2,
                                   minRadius=r_min, maxRadius=r_max)
        detections = []
        if circles is not None:
            for (cx, cy, r) in circles[0]:
                w, l = measure_boundary(gray, cx, cy, r, dark_thresh)
                detections.append((float(cx), float(cy), w, l))

        unmatched = list(range(len(detections)))
        if active and detections:
            pred = np.array([t.predict() for t in active])
            dp_ = np.array([[d[0], d[1]] for d in detections])
            D = dist.cdist(pred, dp_)
            used_t, used_d = set(), set()
            for _ in range(min(len(active), len(detections))):
                i, j = np.unravel_index(np.argmin(D), D.shape)
                if not np.isfinite(D[i, j]) or D[i, j] > MATCH_DIST:
                    break
                cx, cy, w, l = detections[j]
                active[i].update(cx, cy, w, l, fi)
                used_t.add(i); used_d.add(j)
                D[i, :] = np.inf; D[:, j] = np.inf
            unmatched = [j for j in range(len(detections)) if j not in used_d]
            for i, t in enumerate(active):
                if i not in used_t:
                    t.coast()
        else:
            for t in active:
                t.coast()

        for j in unmatched:
            cx, cy, w, l = detections[j]
            active.append(Track(cx, cy, fi, w, l))

        keep = []
        for t in active:
            (all_tracks if t.missing > MAX_MISSING else keep).append(t)
        active = keep

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (0, channel_top), (W, channel_top), (0, 200, 255), 1)
        for t in active:
            if t.missing == 0:
                _, cx, cy, w, l = t.records[-1]
                cx, cy = int(cx), int(cy)
                cv2.rectangle(vis, (cx - int(w/2), cy - int(l/2)),
                              (cx + int(w/2), cy + int(l/2)), (0, 255, 0), 1)
        annotated.append(vis)
    all_tracks.extend(active)
    return all_tracks, annotated


def stitch_tracks(tracks):
    """Greedily merge fragments where one ends near where the next begins,
    a short time later, with little horizontal drift and continuing downward."""
    frags = sorted(tracks, key=lambda t: t.records[0][0])
    used = [False] * len(frags)
    merged = []
    for i in range(len(frags)):
        if used[i]:
            continue
        chain = frags[i]; used[i] = True
        extended = True
        while extended:
            extended = False
            e_fi, e_cx, e_cy, _, _ = chain.records[-1]
            best_j, best_cost = None, None
            for j in range(len(frags)):
                if used[j]:
                    continue
                s_fi, s_cx, s_cy, _, _ = frags[j].records[0]
                dt = s_fi - e_fi
                if dt <= 0 or dt > STITCH_MAX_TIME_GAP:
                    continue
                gap = np.hypot(s_cx - e_cx, s_cy - e_cy)
                if gap > STITCH_MAX_DIST:
                    continue
                if abs(s_cx - e_cx) > STITCH_MAX_X_DRIFT:
                    continue
                if s_cy < e_cy - 5:            # must continue downward (allow slack)
                    continue
                cost = gap + dt
                if best_cost is None or cost < best_cost:
                    best_cost, best_j = cost, j
            if best_j is not None:
                chain.records.extend(frags[best_j].records)
                chain.records.sort(key=lambda r: r[0])
                used[best_j] = True
                extended = True
        merged.append(chain)
    return merged


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------
def main():
    path = get_input_path()
    print(f"Loading: {path}")
    frames, fps = load_frames(path)
    if not frames:
        raise RuntimeError("No frames read.")
    H, W = frames[0].shape
    print(f"{len(frames)} frames | {W}x{H}px | fps={fps:.1f}")

    channel_top = CHANNEL_TOP_Y if CHANNEL_TOP_Y is not None else int(H * 0.40)
    min_travel = (MIN_VERTICAL_TRAVEL if MIN_VERTICAL_TRAVEL is not None
                  else int(H * 0.25))
    r_min, r_max = estimate_radius_range(frames)
    dark_thresh = np.percentile(frames[len(frames)//2], DARK_PERCENTILE)

    raw_tracks, annotated = run_tracking(frames, r_min, r_max,
                                         dark_thresh, channel_top)
    print(f"Raw fragments: {len(raw_tracks)}")

    stitched = stitch_tracks(raw_tracks)
    print(f"After stitching: {len(stitched)}")

    # keep only real spheres: long enough AND travel far enough down
    real = []
    for t in stitched:
        ys = [r[2] for r in t.records]
        if len(t.records) >= MIN_TRANSIT_FRAMES and (max(ys) - min(ys)) >= min_travel:
            real.append(t)
    print(f"Real spheres kept: {len(real)}")
    if not real:
        raise RuntimeError("No real spheres survived. Loosen MIN_TRANSIT_FRAMES "
                           "/ MIN_VERTICAL_TRAVEL or check detection in the video.")

    # label 1..N by entry order (first frame; tie-break by starting y)
    real.sort(key=lambda t: (t.records[0][0], t.records[0][2]))
    for label, t in enumerate(real, start=1):
        t.label = label

    # unit handling
    unit = "um" if UM_PER_PX else "px"
    scale = UM_PER_PX if UM_PER_PX else 1.0

    # ---- per-sphere deformation in X and Y ----
    rows, dim_summary = [], []
    for t in real:
        pre = [(w, l) for (fi, cx, cy, w, l) in t.records if cy < channel_top]
        if len(pre) >= BASELINE_N:
            base_x = np.mean([p[0] for p in pre[:BASELINE_N]])
            base_y = np.mean([p[1] for p in pre[:BASELINE_N]])
            baseline_source = "pre-channel"
        else:
            base_x = t.records[0][3]
            base_y = t.records[0][4]
            baseline_source = "first-seen (no pre-channel frames)"
        if base_x <= 0 or base_y <= 0:
            continue

        widths  = [r[3] for r in t.records]
        lengths = [r[4] for r in t.records]

        for (fi, cx, cy, w, l) in sorted(t.records, key=lambda r: r[0]):
            rows.append({
                "sphere": t.label,
                "frame": fi, "time_s": fi / fps,
                "cx": cx, "cy": cy,
                f"x_dimension_{unit}": w * scale,
                f"y_dimension_{unit}": l * scale,
                "deformation_x_pct": (w - base_x) / base_x * 100.0,
                "deformation_y_pct": (l - base_y) / base_y * 100.0,
                "in_channel": cy >= channel_top,
            })

        dim_summary.append({
            "sphere": t.label,
            "baseline_source": baseline_source,
            f"baseline_x_{unit}": round(base_x * scale, 3),
            f"baseline_y_{unit}": round(base_y * scale, 3),
            "baseline_aspect_ratio_y_over_x": round(base_y / base_x, 3),
            f"equivalent_diameter_{unit}": round((base_x + base_y) / 2 * scale, 3),
            f"min_x_{unit}": round(min(widths) * scale, 3),
            f"max_x_{unit}": round(max(widths) * scale, 3),
            f"min_y_{unit}": round(min(lengths) * scale, 3),
            f"max_y_{unit}": round(max(lengths) * scale, 3),
            "max_deformation_x_pct": round((min(widths) - base_x) / base_x * 100, 2),
            "max_deformation_y_pct": round((max(lengths) - base_y) / base_y * 100, 2),
            "n_frames_tracked": len(t.records),
            "first_frame": t.records[0][0],
            "last_frame": t.records[-1][0],
            "transit_time_s": round((t.records[-1][0] - t.records[0][0]) / fps, 3),
        })

    df = pd.DataFrame(rows)
    dim_df = pd.DataFrame(dim_summary).sort_values("sphere")
    if df.empty:
        raise RuntimeError("No spheres with valid dimensions after filtering.")
    print(f"Final spheres: {sorted(df['sphere'].unique())}")
    print(f"\n=== Particle dimensions studied ({unit}) ===")
    print(dim_df.to_string(index=False))

    # ---- outputs ----
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        desktop = os.path.expanduser("~")
    base = os.path.splitext(os.path.basename(path))[0]
    csv_path  = os.path.join(desktop, f"{base}_deformation.csv")
    dim_path  = os.path.join(desktop, f"{base}_dimensions.csv")
    png_path  = os.path.join(desktop, f"{base}_deformation.png")
    grid_path = os.path.join(desktop, f"{base}_per_sphere.png")
    vid_path  = os.path.join(desktop, f"{base}_annotated.mp4")
    df.to_csv(csv_path, index=False)
    dim_df.to_csv(dim_path, index=False)

    if annotated:
        out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (W, H))
        for v in annotated:
            out.write(v)
        out.release()

    ids = sorted(df["sphere"].unique())
    cmap = plt.get_cmap("tab10")

    # combined graph
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for pid in ids:
        g = df[df.sphere == pid].sort_values("time_s")
        c = cmap((pid - 1) % 10)
        ax.plot(g["time_s"], g["deformation_x_pct"], color=c, lw=1.6, ls="-")
        ax.plot(g["time_s"], g["deformation_y_pct"], color=c, lw=1.6, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Deformation (%)   [(current - baseline)/baseline]")
    ax.set_title("Per-sphere deformation in X and Y through the channel")
    id_handles = [Line2D([0], [0], color=cmap((pid-1) % 10), lw=2,
                         label=f"Sphere {pid}") for pid in ids]
    style_handles = [
        Line2D([0], [0], color="k", lw=1.6, ls="-",  label="Deformation X (width)"),
        Line2D([0], [0], color="k", lw=1.6, ls="--", label="Deformation Y (length)")]
    leg1 = ax.legend(handles=id_handles, title="Sphere",
                     bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    # per-sphere grid
    n = len(ids); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig2, axes = plt.subplots(nrow, ncol, figsize=(3.2*ncol, 2.6*nrow),
                              squeeze=False)
    for k, pid in enumerate(ids):
        a = axes[k // ncol][k % ncol]
        g = df[df.sphere == pid].sort_values("time_s")
        a.plot(g["time_s"], g["deformation_x_pct"], color="tab:blue",
               lw=1.4, label="Deformation X")
        a.plot(g["time_s"], g["deformation_y_pct"], color="tab:red",
               lw=1.4, label="Deformation Y")
        a.axhline(0, color="k", lw=0.6)
        a.set_title(f"Sphere {pid}", fontsize=9)
        a.grid(True, alpha=0.3); a.tick_params(labelsize=7)
    for k in range(n, nrow*ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig2.supxlabel("Time (s)"); fig2.supylabel("Deformation (%)")
    fig2.tight_layout()
    fig2.savefig(grid_path, dpi=200)

    print(f"\nSaved:\n  {csv_path}\n  {dim_path}\n  {png_path}"
          f"\n  {grid_path}\n  {vid_path}")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
