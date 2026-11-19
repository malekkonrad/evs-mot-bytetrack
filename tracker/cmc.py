"""Camera Motion Compensation (CMC) — estymacja affine pomiędzy sąsiednimi
klatkami i korekcja stanu KF.

Algorytm jak w BoT-SORT (Aharon et al. 2022, "Global Motion Compensation"):
  1) Detekcja keypointów: Shi-Tomasi (cv2.goodFeaturesToTrack).
  2) Sparse optical flow: Lucas-Kanade pomiędzy poprzednią a obecną klatką.
  3) Maskowanie keypointów wpadających w bb tracków → ignorujemy ruch dynamiczny.
  4) RANSAC: cv2.estimateAffinePartial2D → macierz scale-rot-translacji 2x3.

Korekcja stanu KF (8D): blockowy kron R z 4 powtórzeniami + translacja na (cx,cy):
  mean[0:2] = R @ mean[0:2] + t
  mean[2:4] = R @ mean[2:4]   # (w, h) — przybliżenie (małe rotacje)
  mean[4:6] = R @ mean[4:6]   # (vx, vy)
  mean[6:8] = R @ mean[6:8]   # (vw, vh)
  cov       = R8 @ cov @ R8.T
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CMC:
    def __init__(
        self,
        max_corners: int = 1000,
        quality_level: float = 0.01,
        min_distance: int = 10,
        ransac_thresh: float = 3.0,
        downscale: float = 1.0,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_thresh = ransac_thresh
        self.downscale = downscale
        self._prev_gray: np.ndarray | None = None
        self._scale_to_orig = 1.0

    def reset(self) -> None:
        self._prev_gray = None

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1.0:
            new_w = int(gray.shape[1] / self.downscale)
            new_h = int(gray.shape[0] / self.downscale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def estimate(self, frame_bgr: np.ndarray, ignore_boxes_xywh: np.ndarray | None = None) -> np.ndarray:
        """Zwraca affine 2x3 mapujące prev_frame → current_frame (w pełnej rozdz.)."""
        gray = self._to_gray(frame_bgr)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)

        if self._prev_gray is None:
            self._prev_gray = gray
            return identity

        # Maska wykluczająca regiony z dynamicznymi obiektami
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if ignore_boxes_xywh is not None and len(ignore_boxes_xywh) > 0:
            scale = 1.0 / self.downscale if self.downscale > 1.0 else 1.0
            for x, y, w, h in ignore_boxes_xywh:
                x1 = max(0, int(x * scale))
                y1 = max(0, int(y * scale))
                x2 = min(gray.shape[1], int((x + w) * scale))
                y2 = min(gray.shape[0], int((y + h) * scale))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        kp = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        if kp is None or len(kp) < 4:
            self._prev_gray = gray
            return identity

        kp_next, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, kp, None)
        if kp_next is None:
            self._prev_gray = gray
            return identity
        good = status.flatten() == 1
        if good.sum() < 4:
            self._prev_gray = gray
            return identity

        src = kp[good].reshape(-1, 2)
        dst = kp_next[good].reshape(-1, 2)
        affine, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
        )
        self._prev_gray = gray
        if affine is None:
            return identity

        # Skalowanie macierzy do oryginalnej rozdzielczości
        if self.downscale > 1.0:
            affine = affine.copy()
            affine[:, 2] *= self.downscale
        return affine


def affine_to_state8(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rozszerza affine 2x3 → (M8 8x8, T8 8) na potrzeby korekcji stanu KF."""
    R = affine[:2, :2]
    t = affine[:2, 2]
    M8 = np.kron(np.eye(4, dtype=float), R)
    T8 = np.zeros(8, dtype=float)
    T8[0:2] = t
    return M8, T8


def load_frame(seq_dir: Path, frame_id: int, ext: str = ".jpg") -> np.ndarray | None:
    """Wczytuje klatkę o numerze 1-based z folderu seq_dir/img1."""
    img_dir = Path(seq_dir) / "img1"
    candidates = [
        img_dir / f"{frame_id:06d}{ext}",
        img_dir / f"{frame_id:05d}{ext}",
        img_dir / f"{frame_id:04d}{ext}",
    ]
    for c in candidates:
        if c.exists():
            img = cv2.imread(str(c), cv2.IMREAD_COLOR)
            return img
    return None
