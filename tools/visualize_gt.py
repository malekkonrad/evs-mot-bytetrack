#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "evs_mot_public_dataset" / "evs_mot-train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "gt_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rysuje ground truth bboxy na obrazkach z evs_mot-train."
    )
    parser.add_argument(
        "--sequence",
        default="MOT_04",
        help="Nazwa sekwencji, np. MOT_02, MOT_03, MOT_04, MOT_05.",
    )
    parser.add_argument(
        "--mot-file",
        type=Path,
        default=None,
        help="Plik MOT do narysowania. Domyslnie gt/gt.txt sekwencji; "
        "np. results/train/MOT_04.txt, zeby zobaczyc predykcje.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Katalog, do ktorego zostana zapisane obrazy podgladowe.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="Pierwsza klatka do zapisania.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ostatnia klatka do zapisania. Domyslnie do konca sekwencji.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Zapisuj co n-ta klatke.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maksymalna liczba zapisanych klatek po uwzglednieniu kroku.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Nie rysuj numerow ID nad bboxami.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Grubosc obramowania bboxa.",
    )
    return parser.parse_args()


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 256,
        (17 * track_id + 80) % 256,
        (29 * track_id + 160) % 256,
    )


def load_gt(gt_path: Path) -> dict[int, list[tuple[int, int, int, int, int]]]:
    frames: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = int(round(float(parts[2])))
            y = int(round(float(parts[3])))
            w = int(round(float(parts[4])))
            h = int(round(float(parts[5])))
            frames[frame_id].append((track_id, x, y, w, h))
    return frames


def draw_annotations(
    image_path: Path,
    annotations: list[tuple[int, int, int, int, int]],
    show_labels: bool,
    thickness: int,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Nie mozna wczytac obrazu: {image_path}")

    for track_id, x, y, w, h in annotations:
        color = color_for_track(track_id)
        x2 = x + w
        y2 = y + h
        cv2.rectangle(image, (x, y), (x2, y2), color, thickness)
        if show_labels:
            label = f"id={track_id}"
            anchor_y = max(20, y - 8)
            cv2.putText(
                image,
                label,
                (x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return image


def main() -> int:
    args = parse_args()

    sequence_dir = DATASET_ROOT / args.sequence
    gt_path = args.mot_file if args.mot_file is not None else sequence_dir / "gt" / "gt.txt"
    img_dir = sequence_dir / "img1"

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono sekwencji: {sequence_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Nie znaleziono gt.txt: {gt_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu z obrazami: {img_dir}")

    gt_by_frame = load_gt(gt_path)
    output_dir = args.output_dir / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"Brak obrazow jpg w: {img_dir}")

    saved = 0
    for image_path in image_paths:
        frame_id = int(image_path.stem)
        if frame_id < args.start_frame:
            continue
        if args.end_frame is not None and frame_id > args.end_frame:
            continue
        if (frame_id - args.start_frame) % args.step != 0:
            continue
        if args.max_frames is not None and saved >= args.max_frames:
            break

        image = draw_annotations(
            image_path=image_path,
            annotations=gt_by_frame.get(frame_id, []),
            show_labels=not args.hide_labels,
            thickness=args.thickness,
        )
        destination = output_dir / image_path.name
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Nie udalo sie zapisac obrazu: {destination}")
        saved += 1

    print(f"Zapisano {saved} klatek do: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
