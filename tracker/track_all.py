"""Uruchamia ByteTracker na wszystkich sekwencjach w danym katalogu.

Standardowo: wynikiem dla MOT_XX jest plik <output_dir>/MOT_XX.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_tracker import run_tracker
from interpolate import interpolate_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteTracker na wszystkich sekwencjach w katalogu.")
    p.add_argument("dataset_dir", type=Path, help="Katalog zawierajacy podkatalogi MOT_XX.")
    p.add_argument("output_dir", type=Path, help="Katalog na pliki wynikowe.")
    p.add_argument("--track-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--new-track-thresh", type=float, default=0.6)
    p.add_argument("--low-thresh", type=float, default=0.1)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--predict-lost-frames", type=int, default=0)
    p.add_argument("--min-box-area", type=float, default=0.0)
    p.add_argument("--lost-velocity-damping", type=float, default=1.0)
    p.add_argument("--no-score-fusion", action="store_true")
    p.add_argument("--cmc", action="store_true")
    p.add_argument("--cmc-downscale", type=float, default=2.0)
    p.add_argument("--interpolate", action="store_true",
                   help="Liniowa interpolacja luk w trackach (post-processing).")
    p.add_argument("--max-gap", type=int, default=20,
                   help="Max dlugosc luki do interpolacji (klatki).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seq_dirs = sorted(p for p in args.dataset_dir.iterdir() if p.is_dir())
    if not seq_dirs:
        print(f"Brak sekwencji w {args.dataset_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seq_dir in seq_dirs:
        out_path = args.output_dir / f"{seq_dir.name}.txt"
        n = run_tracker(
            seq_dir,
            out_path,
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
        if args.interpolate:
            n_in, n_out = interpolate_file(out_path, out_path, args.max_gap)
            print(f"  {seq_dir.name}: {n} -> {n_out} (+{n_out - n_in} interp) -> {out_path}")
        else:
            print(f"  {seq_dir.name}: {n} linii -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
