"""Funkcje pomocnicze: wektoryzowane IoU + przypisanie Hungarian."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU pomiędzy zestawami bb w formacie tlwh ([x, y, w, h]).

    Zwraca macierz (len(a), len(b)). Puste zestawy → macierz zer.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    a_x1 = a[:, 0][:, None]
    a_y1 = a[:, 1][:, None]
    a_x2 = a_x1 + a[:, 2][:, None]
    a_y2 = a_y1 + a[:, 3][:, None]
    b_x1 = b[:, 0][None, :]
    b_y1 = b[:, 1][None, :]
    b_x2 = b_x1 + b[:, 2][None, :]
    b_y2 = b_y1 + b[:, 3][None, :]

    inter_w = np.clip(np.minimum(a_x2, b_x2) - np.maximum(a_x1, b_x1), 0, None)
    inter_h = np.clip(np.minimum(a_y2, b_y2) - np.maximum(a_y1, b_y1), 0, None)
    inter = inter_w * inter_h
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def iou_distance(tracks: list, detections_xywh: np.ndarray) -> np.ndarray:
    """Macierz kosztów 1 - IoU (mniejsze = lepsze)."""
    if len(tracks) == 0 or len(detections_xywh) == 0:
        return np.zeros((len(tracks), len(detections_xywh)), dtype=float)
    track_boxes = np.asarray([t.tlwh for t in tracks], dtype=float)
    return 1.0 - iou_matrix(track_boxes, detections_xywh)


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fuzja score'ów z IoU (z ByteTrack: similarity *= score)."""
    if cost.size == 0:
        return cost
    similarity = 1.0 - cost
    fused = similarity * scores[None, :]
    return 1.0 - fused


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian z progiem kosztu. Pary z kosztem > thresh są odrzucane.

    Zwraca: (matches, unmatched_a, unmatched_b).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    cost = cost.copy()
    cost[cost > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_a = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_b = [j for j in range(cost.shape[1]) if j not in matched_cols]
    return matches, unmatched_a, unmatched_b
