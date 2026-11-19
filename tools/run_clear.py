#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np


TRACK_EVAL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "TrackEval"
if str(TRACK_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_EVAL_ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from trackeval.metrics.clear import CLEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uruchamia metryke CLEAR z TrackEval na plikach w formacie MOT."
    )
    parser.add_argument("gt", type=Path, help="Sciezka do pliku ground truth w formacie MOT.")
    parser.add_argument("tracker", type=Path, help="Sciezka do pliku wynikowego trackera w formacie MOT.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prog podobienstwa IoU przekazywany do CLEAR (domyslnie: 0.5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Wypisz wynik jako JSON zamiast krotkiej tabeli.",
    )
    return parser.parse_args()


def load_mot_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"Za malo kolumn w pliku {path}: {line}")
            rows.append([float(value) for value in parts[:6]])

    if not rows:
        return np.empty((0, 6), dtype=float)
    return np.asarray(rows, dtype=float)


def iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt_x1 = gt_boxes[:, 0][:, None]
    gt_y1 = gt_boxes[:, 1][:, None]
    gt_x2 = gt_x1 + gt_boxes[:, 2][:, None]
    gt_y2 = gt_y1 + gt_boxes[:, 3][:, None]

    tr_x1 = tracker_boxes[:, 0][None, :]
    tr_y1 = tracker_boxes[:, 1][None, :]
    tr_x2 = tr_x1 + tracker_boxes[:, 2][None, :]
    tr_y2 = tr_y1 + tracker_boxes[:, 3][None, :]

    inter_x1 = np.maximum(gt_x1, tr_x1)
    inter_y1 = np.maximum(gt_y1, tr_y1)
    inter_x2 = np.minimum(gt_x2, tr_x2)
    inter_y2 = np.minimum(gt_y2, tr_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = gt_boxes[:, 2][:, None] * gt_boxes[:, 3][:, None]
    tr_area = tracker_boxes[:, 2][None, :] * tracker_boxes[:, 3][None, :]
    union = gt_area + tr_area - inter_area
    return np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)


def build_trackeval_data(gt_rows: np.ndarray, tracker_rows: np.ndarray) -> dict[str, object]:
    gt_id_values = sorted({int(row[1]) for row in gt_rows})
    tracker_id_values = sorted({int(row[1]) for row in tracker_rows})
    gt_id_map = {track_id: index for index, track_id in enumerate(gt_id_values)}
    tracker_id_map = {track_id: index for index, track_id in enumerate(tracker_id_values)}

    max_frame = 0
    if len(gt_rows):
        max_frame = max(max_frame, int(np.max(gt_rows[:, 0])))
    if len(tracker_rows):
        max_frame = max(max_frame, int(np.max(tracker_rows[:, 0])))

    gt_ids = []
    tracker_ids = []
    similarity_scores = []

    for frame in range(1, max_frame + 1):
        gt_frame = gt_rows[gt_rows[:, 0] == frame]
        tracker_frame = tracker_rows[tracker_rows[:, 0] == frame]

        gt_frame_ids = np.asarray([gt_id_map[int(row[1])] for row in gt_frame], dtype=int)
        tracker_frame_ids = np.asarray([tracker_id_map[int(row[1])] for row in tracker_frame], dtype=int)

        gt_boxes = gt_frame[:, 2:6] if len(gt_frame) else np.empty((0, 4), dtype=float)
        tracker_boxes = tracker_frame[:, 2:6] if len(tracker_frame) else np.empty((0, 4), dtype=float)

        gt_ids.append(gt_frame_ids)
        tracker_ids.append(tracker_frame_ids)
        similarity_scores.append(iou_matrix(gt_boxes, tracker_boxes))

    return {
        "num_timesteps": max_frame,
        "num_gt_ids": len(gt_id_values),
        "num_tracker_ids": len(tracker_id_values),
        "num_gt_dets": len(gt_rows),
        "num_tracker_dets": len(tracker_rows),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def print_table(result: dict[str, float | int]) -> None:
    fields = [
        "MOTA",
        "MOTP",
        "CLR_Re",
        "CLR_Pr",
        "CLR_F1",
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "Frag",
    ]
    width = max(len(field) for field in fields)
    for field in fields:
        value = result[field]
        if isinstance(value, (float, np.floating)):
            print(f"{field:<{width}} : {value:.6f}")
        else:
            print(f"{field:<{width}} : {int(value)}")


def main() -> int:
    args = parse_args()

    gt_rows = load_mot_file(args.gt)
    tracker_rows = load_mot_file(args.tracker)
    data = build_trackeval_data(gt_rows, tracker_rows)

    metric = CLEAR(config={"THRESHOLD": args.threshold, "PRINT_CONFIG": False})
    result = metric.eval_sequence(data)

    if args.json:
        serializable = {
            key: float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value
            for key, value in result.items()
        }
        print(json.dumps(serializable, indent=2))
    else:
        print_table(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
