"""Uruchamia ByteTracker na zbiorze treningowym i liczy metryki CLEAR.

Wartości MOTA/MOTP/IDSW per sekwencja oraz agregat (suma TP/FN/FP/IDSW).
Pomocne do strojenia hiperparametrów. Przyjmuje wszystkie te same flagi co
track_all.py + ścieżki do datasetu treningowego oraz katalog tymczasowy
na wyniki.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from run_tracker import run_tracker
from interpolate import interpolate_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_EVAL_ROOT = REPO_ROOT / "third_party" / "TrackEval"
if str(TRACK_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_EVAL_ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from trackeval.metrics.clear import CLEAR  # type: ignore

# run_clear.py mieszka w tools/
sys.path.insert(0, str(REPO_ROOT / "tools"))
from run_clear import build_trackeval_data, load_mot_file  # type: ignore


def evaluate_pair(gt_path: Path, tracker_path: Path, threshold: float = 0.5) -> dict:
    gt_rows = load_mot_file(gt_path)
    tr_rows = load_mot_file(tracker_path)
    data = build_trackeval_data(gt_rows, tr_rows)
    metric = CLEAR(config={"THRESHOLD": threshold, "PRINT_CONFIG": False})
    return metric.eval_sequence(data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ByteTracker + CLEAR na sekwencjach treningowych.")
    p.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "data" / "evs_mot_public_dataset" / "evs_mot-train",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "results" / "train",
    )
    p.add_argument("--track-thresh", type=float, default=0.5)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--new-track-thresh", type=float, default=0.6)
    p.add_argument("--low-thresh", type=float, default=0.1)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--predict-lost-frames", type=int, default=0)
    p.add_argument("--min-box-area", type=float, default=0.0)
    p.add_argument("--lost-velocity-damping", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5, help="IoU prog dla CLEAR.")
    p.add_argument("--no-score-fusion", action="store_true")
    p.add_argument("--cmc", action="store_true")
    p.add_argument("--cmc-downscale", type=float, default=2.0)
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--max-gap", type=int, default=20)
    return p.parse_args()


def fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):8.3f}"
    return f"{int(value):8d}"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted(p for p in args.dataset_dir.iterdir() if p.is_dir())

    fields = ["MOTA", "MOTP", "CLR_Re", "CLR_Pr", "CLR_F1", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag"]
    header = f"{'sekwencja':<10}" + "".join(f"{f:>9}" for f in fields)
    print(header)
    print("-" * len(header))

    totals = {k: 0 for k in ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag"]}
    motp_num = 0.0
    motp_den = 0
    for seq_dir in seq_dirs:
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        out_path = args.output_dir / f"{seq_dir.name}.txt"
        run_tracker(
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
            interpolate_file(out_path, out_path, args.max_gap)
        result = evaluate_pair(gt_path, out_path, threshold=args.threshold)
        row = f"{seq_dir.name:<10}" + "".join(fmt(result[f]) for f in fields)
        print(row)

        for k in totals:
            totals[k] += int(result[k])
        # MOTP_sum/CLR_TP zachowane separately
        motp_num += float(result["MOTP"]) * float(result["CLR_TP"])
        motp_den += int(result["CLR_TP"])

    total_gt = totals["CLR_TP"] + totals["CLR_FN"]
    if total_gt > 0:
        agg_mota = 1.0 - (totals["CLR_FN"] + totals["CLR_FP"] + totals["IDSW"]) / total_gt
    else:
        agg_mota = float("nan")
    agg_motp = (motp_num / motp_den) if motp_den > 0 else float("nan")
    agg_re = totals["CLR_TP"] / total_gt if total_gt > 0 else float("nan")
    agg_pr = totals["CLR_TP"] / (totals["CLR_TP"] + totals["CLR_FP"]) if (totals["CLR_TP"] + totals["CLR_FP"]) > 0 else float("nan")

    print("-" * len(header))
    print(
        f"{'AGG':<10}"
        f"{fmt(agg_mota)}{fmt(agg_motp)}{fmt(agg_re)}{fmt(agg_pr)}"
        f"{'':>9}"
        f"{fmt(totals['CLR_TP'])}{fmt(totals['CLR_FN'])}{fmt(totals['CLR_FP'])}"
        f"{fmt(totals['IDSW'])}{fmt(totals['Frag'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
