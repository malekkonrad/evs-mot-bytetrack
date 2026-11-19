"""Uruchamia ByteTracker na pojedynczej sekwencji w formacie MOTChallenge.

Wczytuje:
  <seq_dir>/det/det.txt           — detekcje
  <seq_dir>/seqinfo.ini           — metadane (frameRate, seqLength)

Pisze:
  <output_path>                   — wynik trackera w formacie MOT
                                    `<frame>,<id>,<x>,<y>,<w>,<h>,1,-1,-1,-1`
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np

from byte_tracker import ByteTracker
from cmc import CMC, load_frame


def load_seqinfo(seq_dir: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    section = cfg["Sequence"]
    return {
        "name": section.get("name", seq_dir.name),
        "frame_rate": float(section.get("frameRate", "30")),
        "seq_length": int(section.get("seqLength", "0")),
        "im_width": int(section.get("imWidth", "0")),
        "im_height": int(section.get("imHeight", "0")),
    }


def load_detections(det_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Wczytuje detekcje pogrupowane po klatkach: frame -> (xywh, scores)."""
    if not det_path.exists():
        raise FileNotFoundError(det_path)
    rows = []
    with det_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise ValueError(f"Za malo kolumn w {det_path}: {line}")
            frame = int(float(parts[0]))
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            score = float(parts[6])
            rows.append((frame, x, y, w, h, score))
    if not rows:
        return {}
    by_frame: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for frame, x, y, w, h, score in rows:
        by_frame.setdefault(frame, []).append((x, y, w, h, score))
    out = {}
    for frame, items in by_frame.items():
        arr = np.asarray(items, dtype=float)
        out[frame] = (arr[:, :4], arr[:, 4])
    return out


def run_tracker(seq_dir: Path, output_path: Path, *, use_cmc: bool = False,
                cmc_downscale: float = 2.0, **tracker_kwargs) -> int:
    info = load_seqinfo(seq_dir)
    detections = load_detections(seq_dir / "det" / "det.txt")
    seq_length = info["seq_length"]
    if seq_length <= 0 and detections:
        seq_length = max(detections)

    frame_rate = tracker_kwargs.pop("frame_rate", info["frame_rate"])
    tracker = ByteTracker(frame_rate=frame_rate, **tracker_kwargs)

    cmc: CMC | None = CMC(downscale=cmc_downscale) if use_cmc else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for frame in range(1, seq_length + 1):
            xywh, scores = detections.get(frame, (np.empty((0, 4)), np.empty((0,))))
            if cmc is not None:
                img = load_frame(seq_dir, frame)
                if img is not None:
                    # Maskuj keypointy w obszarach detekcji (dynamiczne obiekty)
                    affine = cmc.estimate(img, ignore_boxes_xywh=xywh)
                    tracker.apply_cmc(affine)
            online_tracks = tracker.update(xywh, scores)
            for t in online_tracks:
                x, y, w, h = t.tlwh
                # Format MOT: frame,id,x,y,w,h,1,-1,-1,-1
                fh.write(
                    f"{frame},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )
                written += 1
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteTracker dla pojedynczej sekwencji MOT.")
    p.add_argument("seq_dir", type=Path, help="Katalog sekwencji (zawierajacy det/, seqinfo.ini).")
    p.add_argument("output", type=Path, help="Sciezka do pliku wynikowego .txt.")
    p.add_argument("--track-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--new-track-thresh", type=float, default=0.6)
    p.add_argument("--low-thresh", type=float, default=0.1)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--predict-lost-frames", type=int, default=0,
                   help="Emituj predykcje Kalmana dla niedawno straconych torów (liczba klatek).")
    p.add_argument("--min-box-area", type=float, default=0.0)
    p.add_argument("--lost-velocity-damping", type=float, default=1.0)
    p.add_argument("--no-score-fusion", action="store_true")
    p.add_argument("--cmc", action="store_true", help="Camera Motion Compensation.")
    p.add_argument("--cmc-downscale", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    written = run_tracker(
        args.seq_dir,
        args.output,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        new_track_thresh=args.new_track_thresh,
        low_thresh=args.low_thresh,
        track_buffer=args.track_buffer,
        predict_lost_frames=args.predict_lost_frames,
        min_box_area=args.min_box_area,
        lost_velocity_damping=args.lost_velocity_damping,
        use_score_fusion=not args.no_score_fusion,
        use_cmc=args.cmc,
        cmc_downscale=args.cmc_downscale,
    )
    print(f"OK: {args.seq_dir.name} -> {args.output} ({written} linii)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
