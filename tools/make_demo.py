"""Renderuje film demo z wynikami trackera (MP4 i/lub GIF).

Rysuje bboxy z ID, ślady trajektorii (dolny środek bboxa) i pasek HUD.
Z `--gt` dokleja obok panel z ground truth (tylko sekwencje treningowe).

    uv run python tools/make_demo.py --sequence MOT_01 \\
        --mot-file results/test/MOT_01.txt --out assets/demo_MOT_01.mp4

    uv run python tools/make_demo.py --sequence MOT_05 \\
        --mot-file results/train/MOT_05.txt --gt \\
        --out assets/compare_MOT_05.gif --width 1280 --fps 12 --end-frame 300

Wymaga ffmpeg w PATH.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "evs_mot_public_dataset"


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Stabilny, nasycony kolor BGR dla danego ID (HSV ze złotym podziałem)."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 179)
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def load_mot(path: Path, is_gt: bool) -> dict[int, list[tuple[int, float, float, float, float]]]:
    frames: dict[int, list] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split(",")
        if len(parts) < 6:
            continue
        if is_gt and len(parts) >= 8:
            # tak samo jak przy ewaluacji: eval_flag == 1 i klasa 1 (ludzie)
            if int(float(parts[6])) != 1 or int(float(parts[7])) != 1:
                continue
        frame, tid = int(float(parts[0])), int(float(parts[1]))
        x, y, w, h = map(float, parts[2:6])
        frames[frame].append((tid, x, y, w, h))
    return frames


def find_seq_dir(name: str) -> Path:
    for split in ("evs_mot-train", "evs_mot-test"):
        d = DATA_ROOT / split / name
        if d.exists():
            return d
    raise FileNotFoundError(f"Nie znaleziono {name} w {DATA_ROOT}")


class Renderer:
    def __init__(self, annotations, title: str, trail: int, scale: float):
        self.ann = annotations
        self.title = title
        self.trail = trail
        self.scale = scale
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail))
        self.seen: set[int] = set()

    def draw(self, img: np.ndarray, frame_id: int) -> np.ndarray:
        s = self.scale
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        thick = max(1, round(3 * s))
        font = 0.9 * s + 0.2
        boxes = self.ann.get(frame_id, [])
        active = set()

        for tid, x, y, w, h in boxes:
            active.add(tid)
            self.seen.add(tid)
            self.history[tid].append((int((x + w / 2) * s), int((y + h) * s)))

        # ślady pod boxami
        overlay = img.copy()
        for tid, pts in self.history.items():
            if tid not in active or len(pts) < 2:
                continue
            cv2.polylines(overlay, [np.array(pts, np.int32)], False,
                          color_for_track(tid), thick, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

        for tid, x, y, w, h in boxes:
            c = color_for_track(tid)
            p1, p2 = (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s))
            cv2.rectangle(img, p1, p2, c, thick, cv2.LINE_AA)
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            ly = max(th + 4, p1[1])
            cv2.rectangle(img, (p1[0], ly - th - 4), (p1[0] + tw + 4, ly), c, -1)
            cv2.putText(img, label, (p1[0] + 2, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        font, (0, 0, 0), 1, cv2.LINE_AA)

        hud = f"{self.title}  |  frame {frame_id}  |  on screen {len(boxes)}  |  IDs so far {len(self.seen)}"
        bar_h = int(18 + 24 * s)
        cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Film demo z wynikami trackera.")
    p.add_argument("--sequence", required=True)
    p.add_argument("--mot-file", type=Path, required=True, help="Wynik trackera w formacie MOT.")
    p.add_argument("--gt", action="store_true", help="Dodaj obok panel z ground truth.")
    p.add_argument("--out", type=Path, required=True, help=".mp4, .gif albo .webp (animowany)")
    p.add_argument("--width", type=int, default=960, help="Szerokość całego filmu w px.")
    p.add_argument("--fps", type=float, default=30.0, help="FPS wyjściowe.")
    p.add_argument("--step", type=int, default=1, help="Bierz co n-tą klatkę.")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--trail", type=int, default=40, help="Długość śladu w klatkach.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seq_dir = find_seq_dir(args.sequence)
    images = sorted((seq_dir / "img1").glob("*.jpg"))
    images = [p for p in images if int(p.stem) >= args.start_frame
              and (args.end_frame is None or int(p.stem) <= args.end_frame)]

    panels = 2 if args.gt else 1
    first = cv2.imread(str(images[0]))
    scale = args.width / panels / first.shape[1]

    renderers = [Renderer(load_mot(args.mot_file, is_gt=False), "ByteTrack + CMC", args.trail, scale)]
    if args.gt:
        gt_path = seq_dir / "gt" / "gt.txt"
        renderers.append(Renderer(load_mot(gt_path, is_gt=True), "Ground truth", args.trail, scale))

    h = int(round(first.shape[0] * scale)) // 2 * 2
    w = int(round(first.shape[1] * scale)) * panels // 2 * 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix == ".gif":
        vf = ("split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        out_opts = ["-vf", vf, "-loop", "0"]
    elif args.out.suffix == ".webp":
        out_opts = ["-c:v", "libwebp_anim", "-quality", "60", "-compression_level", "6",
                    "-loop", "0"]
    else:
        out_opts = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    "-preset", "slow", "-movflags", "+faststart"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-", *out_opts, str(args.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        frame_id = int(path.stem)
        # historia śladów aktualizuje się co klatkę, nawet jeśli klatka nie trafi do filmu
        tiles = [r.draw(img, frame_id) for r in renderers]
        if i % args.step:
            continue
        frame = np.hstack(tiles) if len(tiles) > 1 else tiles[0]
        proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())

    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg zakończył się błędem")
    print(f"Zapisano {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
