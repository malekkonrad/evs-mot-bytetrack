"""Live wizualizator MOT: rysuje bboxy z pliku MOT na klatkach sekwencji.

Uruchamiać dwa razy obok siebie (predykcje vs GT):

    uv run python tools/visualize_live.py \\
        --sequence MOT_05 \\
        --mot-file results/train/MOT_05.txt \\
        --title "Predictions"

    uv run python tools/visualize_live.py \\
        --sequence MOT_05 \\
        --mot-file data/evs_mot_public_dataset/evs_mot-train/MOT_05/gt/gt.txt \\
        --gt --title "Ground Truth"

Klawisze: q=quit, space=pauza, n / →=krok dalej, p / ←=krok wstecz,
r=restart, +/-=zmiana fps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "data" / "evs_mot_public_dataset" / "evs_mot-train"
TEST_ROOT = REPO_ROOT / "data" / "evs_mot_public_dataset" / "evs_mot-test"


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 256,
        (17 * track_id + 80) % 256,
        (29 * track_id + 160) % 256,
    )


def load_mot(
    path: Path,
    is_gt: bool = False,
    gt_class_filter: int = 1,
    gt_eval_flag_only: bool = True,
) -> dict[int, list[tuple[int, int, int, int, int, float]]]:
    """Wczytuje plik MOT. Dla GT filtruje class i eval_flag.

    Zwraca: frame -> [(track_id, x, y, w, h, score_or_visibility), ...]
    """
    frames: dict[int, list] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
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
            extra = 1.0
            if is_gt and len(parts) >= 9:
                eval_flag = int(float(parts[6]))
                cls = int(float(parts[7]))
                visibility = float(parts[8])
                if gt_eval_flag_only and eval_flag != 1:
                    continue
                if gt_class_filter is not None and cls != gt_class_filter:
                    continue
                extra = visibility
            elif not is_gt and len(parts) >= 7:
                try:
                    extra = float(parts[6])
                except ValueError:
                    extra = 1.0
            frames[frame_id].append((track_id, x, y, w, h, extra))
    return frames


def draw(
    image,
    annotations: list,
    show_labels: bool,
    thickness: int,
    show_extra: bool,
):
    for track_id, x, y, w, h, extra in annotations:
        color = color_for_track(track_id)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        if show_labels:
            label = f"{track_id}"
            if show_extra:
                label += f" {extra:.2f}"
            anchor_y = max(15, y - 6)
            cv2.putText(
                image,
                label,
                (x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return image


def overlay_hud(image, frame_id: int, total: int, n_boxes: int, fps: float, paused: bool, title: str):
    h, w = image.shape[:2]
    bar = f"{title} | frame {frame_id}/{total} | boxes {n_boxes} | {fps:.0f} fps {'[PAUSED]' if paused else ''}"
    cv2.rectangle(image, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(image, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live wizualizator MOT (predykcje lub GT).")
    p.add_argument("--sequence", required=True, help="Nazwa sekwencji (MOT_02 itd.)")
    p.add_argument("--mot-file", type=Path, required=True, help="Plik MOT z bboxami.")
    p.add_argument("--title", default="MOT", help="Tytuł okna.")
    p.add_argument("--gt", action="store_true", help="Plik to GT — filtruj po eval_flag i class.")
    p.add_argument("--all-classes", action="store_true", help="Pokazuj wszystkie klasy (nie tylko ludzi).")
    p.add_argument("--ignore-eval-flag", action="store_true", help="Nie filtruj po eval_flag w GT.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Katalog z sekwencjami (jeśli nie standardowy train/test).")
    p.add_argument("--fps", type=float, default=30.0, help="Docelowe fps przy autoplay.")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0, help="Skalowanie okna.")
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--no-extra", action="store_true",
                   help="Nie pokazuj score/visibility obok ID.")
    return p.parse_args()


def find_seq_dir(seq_name: str, override: Path | None) -> Path:
    if override is not None:
        d = override / seq_name
        if d.exists():
            return d
        if override.name == seq_name:
            return override
        raise FileNotFoundError(f"Brak {seq_name} w {override}")
    for root in (TRAIN_ROOT, TEST_ROOT):
        d = root / seq_name
        if d.exists():
            return d
    raise FileNotFoundError(f"Nie znaleziono sekwencji {seq_name} w {TRAIN_ROOT} ani {TEST_ROOT}")


def main() -> int:
    args = parse_args()
    seq_dir = find_seq_dir(args.sequence, args.dataset_root)
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Brak {img_dir}")
    if not args.mot_file.exists():
        raise FileNotFoundError(f"Brak {args.mot_file}")

    annotations = load_mot(
        args.mot_file,
        is_gt=args.gt,
        gt_class_filter=None if args.all_classes else 1,
        gt_eval_flag_only=not args.ignore_eval_flag,
    )

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"Brak jpg w {img_dir}")

    frame_ids = [int(p.stem) for p in image_paths]
    total = len(image_paths)

    idx = max(0, args.start_frame - frame_ids[0])
    idx = min(idx, total - 1)
    paused = False
    fps = args.fps

    win = args.title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        frame_id = frame_ids[idx]
        img = cv2.imread(str(image_paths[idx]))
        if img is None:
            print(f"Blad wczytywania {image_paths[idx]}")
            break
        boxes = annotations.get(frame_id, [])
        draw(img, boxes, show_labels=not args.no_labels,
             thickness=args.thickness, show_extra=not args.no_extra)
        overlay_hud(img, frame_id, frame_ids[-1], len(boxes), fps, paused, args.title)

        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_AREA)

        cv2.imshow(win, img)
        delay = 1 if paused else max(1, int(1000.0 / max(fps, 1.0)))
        key = cv2.waitKey(delay) & 0xFFFF

        if key in (ord("q"), 27):  # q lub ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("n"), 83, 0x270F):  # n lub →
            paused = True
            idx = min(idx + 1, total - 1)
        elif key in (ord("p"), 81):  # p lub ←
            paused = True
            idx = max(idx - 1, 0)
        elif key == ord("r"):
            idx = 0
        elif key in (ord("+"), ord("=")):
            fps = min(fps * 1.5, 240.0)
        elif key in (ord("-"), ord("_")):
            fps = max(fps / 1.5, 1.0)
        elif not paused:
            idx = min(idx + 1, total - 1)
            if idx == total - 1:
                paused = True

    cv2.destroyWindow(win)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
