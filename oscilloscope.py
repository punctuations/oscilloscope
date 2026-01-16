import math
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import argparse
import sys
from typing import Iterable, List, Sequence, Tuple, Any, Literal

import cv2
import numpy as np

from tqdm import tqdm

import scipy.spatial as spatial


def sort_algorithm(points: np.ndarray | Sequence[Tuple[int, int]]) -> np.ndarray:
    """Greedy nearest-neighbor ordering using KDTree (fast approximation).

    This replaces the previous asyncio+re-sorting DFS approach which was extremely slow
    for large point clouds.
    """

    pts = np.asarray(points, dtype=np.int32)
    if pts.size == 0:
        return np.asarray([[0, 0]], dtype=np.int32)
    if pts.shape[0] == 1:
        return pts

    ckd: Any = getattr(spatial, "cKDTree", None)
    tree = ckd(pts) if ckd is not None else spatial.KDTree(pts)

    # Precompute a fixed k-nearest-neighbor table once, then traverse it.
    # This avoids per-step KDTree queries and is much faster on dense edge maps.
    k = min(64, len(pts))
    try:
        _, knn = tree.query(pts, k=k, workers=-1)
    except TypeError:
        _, knn = tree.query(pts, k=k)

    knn = np.atleast_2d(knn).astype(np.int32, copy=False)
    visited = np.zeros(len(pts), dtype=bool)
    order = np.empty(len(pts), dtype=np.int32)

    idx = 0
    visited[idx] = True
    order[0] = idx
    current_idx = idx

    for i in range(1, len(pts)):
        next_idx = -1

        for cand in knn[current_idx]:
            if not visited[cand]:
                next_idx = int(cand)
                break

        if next_idx == -1:
            remaining = np.where(~visited)[0]
            cur = pts[current_idx]
            dist = np.abs(pts[remaining, 0] - cur[0]) + \
                np.abs(pts[remaining, 1] - cur[1])
            next_idx = int(remaining[int(dist.argmin())])

        visited[next_idx] = True
        order[i] = next_idx
        current_idx = next_idx

    return pts[order]


def _morton_split_by_1(a: np.ndarray) -> np.ndarray:
    """Split bits of 16-bit integers so they occupy even bits in 32-bit output."""
    a = a.astype(np.uint32, copy=False)
    a = (a | (a << 8)) & 0x00FF00FF
    a = (a | (a << 4)) & 0x0F0F0F0F
    a = (a | (a << 2)) & 0x33333333
    a = (a | (a << 1)) & 0x55555555
    return a


def sort_morton(points: np.ndarray | Sequence[Tuple[int, int]]) -> np.ndarray:
    """Very fast locality-preserving ordering using Morton (Z-order) codes."""
    pts = np.asarray(points, dtype=np.int32)
    if pts.size == 0:
        return np.asarray([[0, 0]], dtype=np.int32)
    if pts.shape[0] == 1:
        return pts

    # Clamp to 16-bit to keep bit-interleaving simple.
    x = np.clip(pts[:, 0], 0, 65535).astype(np.uint16, copy=False)
    y = np.clip(pts[:, 1], 0, 65535).astype(np.uint16, copy=False)
    code = _morton_split_by_1(x) | (_morton_split_by_1(y) << 1)

    idx = np.argsort(code, kind="mergesort")
    return pts[idx]


@dataclass(frozen=True)
class _AudioConfig:
    sample_rate: int = 44100
    channels: int = 2
    sampwidth_bytes: int = 2  # 16-bit PCM


class _WavStreamWriter:
    def __init__(self, out_path: str, config: _AudioConfig):
        self._out_path = out_path
        self._config = config
        self._wf = wave.open(out_path, 'wb')
        self._wf.setnchannels(config.channels)
        self._wf.setsampwidth(config.sampwidth_bytes)
        self._wf.setframerate(config.sample_rate)

    def write_pcm16_bytes(self, pcm16_bytes: bytes) -> None:
        if not pcm16_bytes:
            return
        self._wf.writeframes(pcm16_bytes)

    def close(self) -> None:
        self._wf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class Oscilloscope:
    def __init__(
        self,
        media_path: str,
        hide_progress: bool = False,
        is_image: bool = False,
        duration_s: int = 1,
        *,
        scale: float = 0.5,
        max_points_per_frame: int = 40000,
        workers: int = 0,
        prefetch: int = 0,
        use_contours: bool = False,
        blur_ksize: int = 3,
        skip_solid_frames: bool = True,
        solid_threshold: int = 2,
        ordering: Literal["morton", "kd", "kd-flat"] = "kd",
        fast: bool = False,
    ):
        self.scale = float(scale)
        if not (0.05 <= self.scale <= 1.0):
            raise ValueError("scale must be between 0.05 and 1.0")

        # Open the media file
        if is_image:
            self.image = cv2.imread(media_path)
            # Check if the video file was opened successfully
            if not (type(self.image) is np.ndarray):
                raise IOError("Failed to open the image file.")

            self.frame_width = int(self.image.shape[1] * self.scale)
            self.frame_height = int(self.image.shape[0] * self.scale)
            self.duration = duration_s
        else:
            self.video = cv2.VideoCapture(media_path)
            # Check if the video file was opened successfully
            if not self.video.isOpened():
                raise IOError("Failed to open the video file.")

            # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            fps = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video.get(
                cv2.CAP_PROP_FRAME_WIDTH) * self.scale)  # reduced width
            self.frame_height = int(self.video.get(
                cv2.CAP_PROP_FRAME_HEIGHT) * self.scale)  # reduced height
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            # duration per frame (used for speeding up playback of audio)
            self.frame_duration = 1 / fps

        # flags
        self.is_image: bool = is_image
        self.hide_progress: bool = hide_progress

        # perf knobs
        self.max_points_per_frame = int(max_points_per_frame)
        self.workers = int(workers)
        self.prefetch = int(prefetch)
        self.use_contours = bool(use_contours)
        self.ordering: Literal["morton", "kd", "kd-flat"] = ordering

        # noise reduction
        self.blur_ksize = int(blur_ksize)

        # solid-frame skip (fast path)
        self.skip_solid_frames = bool(skip_solid_frames)
        self.solid_threshold = int(solid_threshold)

        # fast path: skip expensive binarization/background normalization
        self.fast = bool(fast)

        # cached per-frame constants
        self._dsize = (self.frame_width, self.frame_height)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        self._small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._solid_check_dsize = (48, 48)

        # Internal defaults for "good in-between" output quality.
        # Goal: fewer noisy speckles + fewer long connecting lines between disjoint shapes.
        self._contour_min_area = 80.0
        self._contour_min_points = 40
        self._contour_max_count = 6
        self._border_margin_px = 2
        self._border_area_frac = 0.20

        # Instead of aggressively simplifying geometry, keep detail and resample uniformly.
        self._contour_resample_closed = True

        # In no-contours mode, treat connected edge components as separate shapes.
        # This avoids the sorter interleaving disconnected shapes (which creates faint jump lines).
        self._component_min_pixels = 30
        self._component_max_count = 40

        self._audio = _AudioConfig(sample_rate=44100)

    def convert(self, out_path: str = 'out.wav'):
        # stream-write WAV output (avoids O(n^2) reread/concat/rewrite)

        with _WavStreamWriter(out_path, self._audio) as writer:
            if self.is_image:
                pcm = self.__frame_to_pcm16_bytes(self.image, is_image=True)
                writer.write_pcm16_bytes(pcm)
                return

            # Video: read frames sequentially, process concurrently, write in-order.
            max_workers = self.workers if self.workers > 0 else max(
                1, (os.cpu_count() or 4) - 1)
            prefetch = self.prefetch if self.prefetch > 0 else max_workers * 2

            def submit_one(executor):
                eof, frame = self.video.read()
                if not eof:
                    return None
                return executor.submit(self.__frame_to_pcm16_bytes, frame, False)

            written = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                # prime the pipeline
                while len(futures) < prefetch and self.video.isOpened():
                    fut = submit_one(executor)
                    if fut is None:
                        break
                    futures.append(fut)

                if self.hide_progress:
                    while futures:
                        pcm = futures.pop(0).result()
                        writer.write_pcm16_bytes(pcm)
                        written += 1

                        fut = submit_one(executor)
                        if fut is not None:
                            futures.append(fut)
                else:
                    with tqdm(total=self.frame_count, unit="frames") as pbar:
                        while futures:
                            pcm = futures.pop(0).result()
                            writer.write_pcm16_bytes(pcm)
                            written += 1
                            pbar.update(1)

                            fut = submit_one(executor)
                            if fut is not None:
                                futures.append(fut)

            self.video.release()

    def __detect_path(self, frame):
        # resize image (INTER_AREA is typically best when shrinking)
        frame_resized = cv2.resize(
            frame, self._dsize, interpolation=cv2.INTER_AREA)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        if self.fast:
            # Fast mode: edge detection directly from grayscale.
            if self.blur_ksize and self.blur_ksize >= 3:
                k = self.blur_ksize
                if k % 2 == 0:
                    k += 1
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            edges = cv2.Canny(gray, 80, 160)

            # Truly fast path: trace contours directly from the edge map.
            # This preserves shape integrity (no interleaving) and avoids the expensive KD ordering.
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return np.asarray([[0, 0]], dtype=np.int32)

            # Filter tiny contours and keep a bounded amount of work.
            fast_contours = [c for c in contours if c.shape[0]
                             >= self._component_min_pixels]
            if not fast_contours:
                return np.asarray([[0, 0]], dtype=np.int32)

            fast_contours.sort(key=lambda c: c.shape[0], reverse=True)
            fast_contours = fast_contours[: self._component_max_count]

            polylines: List[np.ndarray] = [
                c.reshape(-1, 2).astype(np.int32, copy=False) for c in fast_contours]

            # Optional global downsample to respect the point budget.
            if self.max_points_per_frame > 0:
                total_pts = sum(int(p.shape[0]) for p in polylines)
                if total_pts > self.max_points_per_frame:
                    step = int(
                        math.ceil(total_pts / self.max_points_per_frame))
                    polylines = [p[::step]
                                 for p in polylines if p.shape[0] > 0]

            # Chain by nearest endpoints to minimize travel lines.
            remaining = list(range(len(polylines)))
            current_i = remaining.pop(0)
            out = polylines[current_i]
            last = out[-1]

            while remaining:
                best_dist = 1_000_000_000
                best_i = remaining[0]
                best_rev = False
                for i in remaining:
                    comp = polylines[i]
                    a = comp[0]
                    b = comp[-1]
                    d_a = abs(int(a[0]) - int(last[0])) + \
                        abs(int(a[1]) - int(last[1]))
                    d_b = abs(int(b[0]) - int(last[0])) + \
                        abs(int(b[1]) - int(last[1]))
                    if d_a < best_dist or d_b < best_dist:
                        if d_a <= d_b:
                            best_dist = d_a
                            best_i = i
                            best_rev = False
                        else:
                            best_dist = d_b
                            best_i = i
                            best_rev = True

                remaining.remove(best_i)
                nxt = polylines[best_i]
                if best_rev:
                    nxt = nxt[::-1]
                out = np.vstack((out, nxt))
                last = out[-1]

            return out

        # remove noise
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, self._morph_kernel)
        out_gray = cv2.divide(gray, bg, scale=255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

        # Light cleanup to reduce speckle noise before edge detection.
        if self.blur_ksize and self.blur_ksize >= 3:
            k = self.blur_ksize
            if k % 2 == 0:
                k += 1
            out_binary = cv2.GaussianBlur(out_binary, (k, k), 0)

        out_binary = cv2.morphologyEx(
            out_binary, cv2.MORPH_OPEN, self._small_kernel)
        out_binary = cv2.morphologyEx(
            out_binary, cv2.MORPH_CLOSE, self._small_kernel)

        if self.use_contours:
            path: List[Tuple[int, int]] = []
            edges = cv2.Canny(out_binary, 100, 200)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            h, w = out_binary.shape[:2]
            frame_area = float(h * w)

            filtered: List[Tuple[float, np.ndarray]] = []
            for c in contours:
                if c.shape[0] < self._contour_min_points:
                    continue
                area = float(cv2.contourArea(c))
                if area < self._contour_min_area:
                    continue

                # Drop big contours that touch the frame border (commonly causes the "border box" artifact).
                x, y, bw, bh = cv2.boundingRect(c)
                touches = (
                    x <= self._border_margin_px
                    or y <= self._border_margin_px
                    or (x + bw) >= (w - self._border_margin_px)
                    or (y + bh) >= (h - self._border_margin_px)
                )
                if touches and area >= (frame_area * self._border_area_frac):
                    continue

                filtered.append((area, c))

            filtered.sort(key=lambda t: t[0], reverse=True)
            if self._contour_max_count > 0:
                filtered = filtered[: self._contour_max_count]

            chains: List[np.ndarray] = []
            for _, c in filtered:
                pts = c.reshape(-1, 2)
                if pts.shape[0] >= 2:
                    chains.append(pts)

            if chains:
                # Compute total perimeter length to allocate points per contour.
                perims = []
                total_perim = 0.0
                for ch in chains:
                    per = float(cv2.arcLength(ch.reshape(-1, 1, 2),
                                self._contour_resample_closed))
                    perims.append(per)
                    total_perim += per

                target_total = 0
                if self.max_points_per_frame > 0:
                    target_total = min(self.max_points_per_frame, sum(
                        int(ch.shape[0]) for ch in chains))
                    target_total = max(target_total, 2000)
                else:
                    target_total = sum(int(ch.shape[0]) for ch in chains)

                # Allocate target points proportionally, but keep a minimum per contour.
                min_per_contour = 400
                alloc = []
                remaining_budget = max(
                    0, target_total - (min_per_contour * len(chains)))
                for per in perims:
                    if total_perim > 0:
                        extra = int(
                            round((per / total_perim) * remaining_budget))
                    else:
                        extra = int(
                            round(remaining_budget / max(1, len(chains))))
                    alloc.append(min_per_contour + max(0, extra))

                # Normalize allocation to exact budget.
                s = sum(alloc)
                if s > 0 and s != target_total:
                    scale = float(target_total) / float(s)
                    alloc = [max(50, int(round(a * scale))) for a in alloc]

                def resample_polyline(points_xy: np.ndarray, n: int, closed: bool) -> np.ndarray:
                    ptsf = points_xy.astype(np.float32, copy=False)
                    if ptsf.shape[0] == 0:
                        return ptsf
                    if n <= 1:
                        return ptsf[:1]
                    if closed:
                        pts2 = np.vstack([ptsf, ptsf[:1]])
                    else:
                        pts2 = ptsf

                    seg = pts2[1:] - pts2[:-1]
                    seglen = np.sqrt((seg * seg).sum(axis=1))
                    cum = np.concatenate([[0.0], np.cumsum(seglen)])
                    total = float(cum[-1])
                    if total <= 1e-6:
                        out = np.repeat(ptsf[:1], n, axis=0)
                        return out

                    # For closed loops, avoid duplicating the start point.
                    endpoint = False if closed else True
                    t = np.linspace(0.0, total, n, endpoint=endpoint)
                    idx = np.searchsorted(cum, t, side='right') - 1
                    idx = np.clip(idx, 0, len(seglen) - 1)
                    local = (t - cum[idx]) / np.maximum(seglen[idx], 1e-6)
                    out = pts2[idx] + (seg[idx] * local[:, None])
                    return out

                # Chain by nearest endpoints to reduce long jumps.
                remaining = list(range(len(chains)))
                current_i = remaining.pop(0)
                current = resample_polyline(
                    chains[current_i], alloc[current_i], self._contour_resample_closed)
                for x, y in current:
                    path.append((int(x), int(y)))
                last = current[-1]

                while remaining:
                    best = 1_000_000_000
                    best_i = remaining[0]
                    best_rev = False
                    for j in remaining:
                        cand = chains[j]
                        a = cand[0].astype(np.int32)
                        b = cand[-1].astype(np.int32)
                        d_a = abs(int(a[0]) - int(last[0])) + \
                            abs(int(a[1]) - int(last[1]))
                        d_b = abs(int(b[0]) - int(last[0])) + \
                            abs(int(b[1]) - int(last[1]))
                        if d_a < best or d_b < best:
                            if d_a <= d_b:
                                best = d_a
                                best_i = j
                                best_rev = False
                            else:
                                best = d_b
                                best_i = j
                                best_rev = True

                    remaining.remove(best_i)
                    nxt_pts = chains[best_i][::-
                                             1] if best_rev else chains[best_i]
                    nxt = resample_polyline(
                        nxt_pts, alloc[best_i], self._contour_resample_closed)

                    # Add only a tiny jump between contours to keep it dim.
                    start = nxt[0]
                    path.append((int(start[0]), int(start[1])))

                    for x, y in nxt:
                        path.append((int(x), int(y)))
                    last = nxt[-1]

            if path:
                # Contour mode returns an already-ordered path.
                return path

        # Default (no-contours): use edge pixels + ordering.
        edges = cv2.Canny(out_binary, 100, 200)
        h, w = edges.shape[:2]
        if self.ordering == "kd":
            # Connected-component grouping to keep shapes intact.
            binary = (edges > 0).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(
                binary, connectivity=8)

            flat = labels.reshape(-1)
            idxs = np.flatnonzero(flat)
            if idxs.size == 0:
                pts = np.asarray([[0, 0]], dtype=np.int32)
                return sort_algorithm(pts)

            lbls = flat[idxs]
            order = np.argsort(lbls, kind="mergesort")
            idxs = idxs[order]
            lbls = lbls[order]

            split = np.flatnonzero(np.diff(lbls)) + 1
            groups = np.split(idxs, split)

            # Filter + keep largest components first.
            comps: List[np.ndarray] = []
            sizes = []
            for g in groups:
                if g.size < self._component_min_pixels:
                    continue
                sizes.append(int(g.size))
                comps.append(g)

            if not comps:
                pts = np.asarray([[0, 0]], dtype=np.int32)
                return sort_algorithm(pts)

            idx_sort = np.argsort(np.asarray(sizes), kind="mergesort")[::-1]
            if self._component_max_count > 0 and idx_sort.size > self._component_max_count:
                idx_sort = idx_sort[: self._component_max_count]
            comps = [comps[i] for i in idx_sort]
            sizes = [sizes[i] for i in idx_sort]

            total = sum(sizes)
            budget = self.max_points_per_frame if self.max_points_per_frame > 0 else total
            budget = max(1, int(budget))

            # Allocate points per component proportionally, with a small minimum.
            min_per = 60
            per_targets: List[int] = []
            if total > 0:
                for s in sizes:
                    t = int(round((s / total) * budget))
                    per_targets.append(max(min_per, min(s, t)))
            else:
                per_targets = [min_per for _ in sizes]

            # Build ordered polylines per component.
            ordered_components: List[np.ndarray] = []
            for g, t in zip(comps, per_targets):
                ys = (g // w).astype(np.int32, copy=False)
                xs = (g % w).astype(np.int32, copy=False)
                pts = np.column_stack((xs, ys)).astype(np.int32, copy=False)
                if pts.shape[0] > t:
                    step = int(math.ceil(pts.shape[0] / t))
                    pts = pts[::step]
                ordered_components.append(sort_algorithm(pts))

            # Chain components by nearest endpoints to minimize travel lines.
            remaining = list(range(len(ordered_components)))
            current_i = remaining.pop(0)
            out = ordered_components[current_i]
            last = out[-1]

            while remaining:
                best_dist = 1_000_000_000
                best_i = remaining[0]
                best_rev = False
                for i in remaining:
                    comp = ordered_components[i]
                    a = comp[0]
                    b = comp[-1]
                    d_a = abs(int(a[0]) - int(last[0])) + \
                        abs(int(a[1]) - int(last[1]))
                    d_b = abs(int(b[0]) - int(last[0])) + \
                        abs(int(b[1]) - int(last[1]))
                    if d_a < best_dist or d_b < best_dist:
                        if d_a <= d_b:
                            best_dist = d_a
                            best_i = i
                            best_rev = False
                        else:
                            best_dist = d_b
                            best_i = i
                            best_rev = True

                remaining.remove(best_i)
                nxt = ordered_components[best_i]
                if best_rev:
                    nxt = nxt[::-1]
                out = np.vstack((out, nxt))
                last = out[-1]

            return out

        # Flat ordering (previous behavior) or Morton.
        y_indices, x_indices = np.nonzero(edges)
        if x_indices.size == 0:
            pts = np.asarray([[0, 0]], dtype=np.int32)
        else:
            pts = np.column_stack((x_indices, y_indices)
                                  ).astype(np.int32, copy=False)
            if self.max_points_per_frame > 0 and pts.shape[0] > self.max_points_per_frame:
                step = int(math.ceil(pts.shape[0] / self.max_points_per_frame))
                pts = pts[::step]

        if self.ordering == "kd-flat":
            return sort_algorithm(pts)
        return sort_morton(pts)

    def __path_to_stereo_waveform(self, path: Sequence[Tuple[int, int]] | np.ndarray) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((1, 2), dtype=np.float32)

        x = (pts[:, 0] - (self.frame_width / 2.0)) / (self.frame_width / 2.0)
        y = -((pts[:, 1] - (self.frame_height / 2.0)) /
              (self.frame_height / 2.0))
        stereo = np.stack((x, y), axis=1).astype(np.float32, copy=False)
        return stereo

    def __resample_stereo(self, stereo: np.ndarray, target_length: int) -> np.ndarray:
        if target_length <= 0:
            target_length = 1
        if stereo.shape[0] == 0:
            stereo = np.zeros((1, 2), dtype=np.float32)

        indices = np.linspace(
            0, stereo.shape[0], target_length, endpoint=False, dtype=np.int64)
        return stereo[indices]

    def __light_smooth_stereo(self, stereo: np.ndarray) -> np.ndarray:
        # Very light fixed smoothing to reduce jitter without blurring too much.
        w = 5
        if stereo.shape[0] < w:
            return stereo
        kernel = (np.ones(w, dtype=np.float32) / float(w))
        out = stereo.copy()
        out[:, 0] = np.convolve(out[:, 0], kernel, mode='same')
        out[:, 1] = np.convolve(out[:, 1], kernel, mode='same')
        return out

    def __is_solid_frame(self, frame) -> bool:
        if not self.skip_solid_frames:
            return False
        if self.solid_threshold < 0:
            return False

        # Downsample hard to make this check very cheap.
        small = cv2.resize(frame, self._solid_check_dsize,
                           interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3 and small.shape[2] >= 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        min_v, max_v, _, _ = cv2.minMaxLoc(gray)
        return (max_v - min_v) <= float(self.solid_threshold)

    def __frame_to_pcm16_bytes(self, frame, is_image: bool) -> bytes:
        # Fast path: solid frames become silence (preserves timing, skips expensive detection).
        if self.__is_solid_frame(frame):
            if is_image:
                total_len = int(self._audio.sample_rate *
                                max(0, int(self.duration)))
                if total_len <= 0:
                    total_len = int(self._audio.sample_rate * (1 / 60))
            else:
                total_len = int(self._audio.sample_rate * self.frame_duration)
                if total_len <= 0:
                    total_len = 1

            zeros = np.zeros((total_len, 2), dtype=np.int16)
            return zeros.tobytes(order='C')

        path = self.__detect_path(frame)
        stereo = self.__path_to_stereo_waveform(path)

        if is_image:
            # Base: one "frame" worth of samples at 60fps for smoother display.
            base_len = int(self._audio.sample_rate * (1 / 60))
            base = self.__resample_stereo(stereo, base_len)
            base = self.__light_smooth_stereo(base)

            total_len = int(self._audio.sample_rate *
                            max(0, int(self.duration)))
            if total_len <= 0:
                total_len = base.shape[0]

            if base.shape[0] < total_len:
                reps = int(math.ceil(total_len / base.shape[0]))
                frame_data = np.tile(base, (reps, 1))[:total_len]
            else:
                frame_data = self.__resample_stereo(base, total_len)
        else:
            target_length = int(self._audio.sample_rate * self.frame_duration)
            frame_data = self.__resample_stereo(stereo, target_length)
            frame_data = self.__light_smooth_stereo(frame_data)

        # float32 [-1,1] -> int16 PCM
        frame_data = np.clip(frame_data, -1.0, 1.0)
        pcm16 = (frame_data * 32767.0).astype(np.int16, copy=False)
        return pcm16.tobytes(order='C')


def _infer_is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return ext in {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m oscilloscope",
        description="Convert an image/video into XY stereo audio for oscilloscope display.",
    )

    p.add_argument("path", help="Path to image or video file")
    p.add_argument("-o", "--out", default="out.wav",
                   help="Output WAV path (default: out.wav)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--image", action="store_true",
                      help="Force treat input as image")
    mode.add_argument("--video", action="store_true",
                      help="Force treat input as video")

    p.add_argument("--duration", type=int, default=1,
                   help="Image duration in seconds (default: 1)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable progress bar")

    p.add_argument("--scale", type=float, default=0.5,
                   help="Downscale factor for processing (0.05-1.0, default: 0.5)")

    p.add_argument("--max-points", type=int, default=40000,
                   help="Max points per frame (default: 40000)")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker threads for video (0=auto)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="Prefetch frames in flight (0=auto)")

    p.add_argument("--blur", type=int, default=3,
                   help="Gaussian blur kernel size (odd int, 0=off, default: 3)")

    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster preprocessing (skip normalization/Otsu); may increase noise",
    )

    p.add_argument(
        "--ordering",
        choices=("morton", "kd", "kd-flat"),
        default="kd",
        help="Point ordering method for no-contours mode (default: kd)",
    )

    solid = p.add_mutually_exclusive_group()
    solid.add_argument("--skip-solid", dest="skip_solid_frames", action="store_true", default=True,
                       help="Skip path detection for solid-color frames (default)")
    solid.add_argument("--no-skip-solid", dest="skip_solid_frames", action="store_false",
                       help="Disable solid-frame skipping")
    p.add_argument("--solid-threshold", type=int, default=2,
                   help="Max grayscale range (max-min) to treat as solid (default: 2)")

    contours = p.add_mutually_exclusive_group()
    contours.add_argument(
        "--contours",
        dest="use_contours",
        action="store_true",
        default=False,
        help="Use contour-based paths (optional)",
    )
    contours.add_argument(
        "--no-contours",
        dest="use_contours",
        action="store_false",
        help="Disable contours; use edge pixels + ordering (default)",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        parser.error(f"File not found: {args.path}")

    if args.image:
        is_image = True
    elif args.video:
        is_image = False
    else:
        is_image = _infer_is_image(args.path)

    osc = Oscilloscope(
        args.path,
        hide_progress=args.no_progress,
        is_image=is_image,
        duration_s=args.duration,
        scale=args.scale,
        max_points_per_frame=args.max_points,
        workers=args.workers,
        prefetch=args.prefetch,
        use_contours=args.use_contours,
        blur_ksize=args.blur,
        skip_solid_frames=args.skip_solid_frames,
        solid_threshold=args.solid_threshold,
        ordering=args.ordering,
        fast=args.fast,
    )
    osc.convert(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
