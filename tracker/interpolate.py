"""Linear track interpolation — post-processing wypełniający luki w tracku.

Dla każdego unikalnego id sortuje wpisy po klatce. Dla par sąsiednich
obserwacji (frame_a, frame_b) z luką 1 < gap ≤ max_gap, dodaje liniowo
interpolowane bbox dla brakujących klatek frame_a+1..frame_b-1.

Działa tylko na lukach WEWNĄTRZ traku (są obserwacje po obu stronach), więc
generuje praktycznie czyste TP. Stosowane w ByteTrack/StrongSORT jako
standardowy zysk MOTA bez kosztu FP.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict


def interpolate_file(in_path: Path, out_path: Path, max_gap: int = 20) -> tuple[int, int]:
    """Wczytuje plik MOT, interpoluje luki, zapisuje. Zwraca (in_lines, out_lines)."""
    by_id: dict[int, list[tuple[int, float, float, float, float, str]]] = defaultdict(list)
    in_count = 0
    with in_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            in_count += 1
            parts = line.split(",")
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(parts[i]) for i in range(2, 6))
            tail = ",".join(parts[6:]) if len(parts) > 6 else "1,-1,-1,-1"
            by_id[tid].append((frame, x, y, w, h, tail))

    all_rows: list[tuple[int, int, float, float, float, float, str]] = []
    for tid, items in by_id.items():
        items.sort(key=lambda r: r[0])
        for i, item in enumerate(items):
            all_rows.append((item[0], tid, item[1], item[2], item[3], item[4], item[5]))
            if i + 1 >= len(items):
                continue
            nxt = items[i + 1]
            gap = nxt[0] - item[0]
            if gap <= 1 or gap > max_gap:
                continue
            for k in range(1, gap):
                t = k / gap
                xk = item[1] + t * (nxt[1] - item[1])
                yk = item[2] + t * (nxt[2] - item[2])
                wk = item[3] + t * (nxt[3] - item[3])
                hk = item[4] + t * (nxt[4] - item[4])
                all_rows.append((item[0] + k, tid, xk, yk, wk, hk, item[5]))

    all_rows.sort(key=lambda r: (r[0], r[1]))
    with out_path.open("w", encoding="utf-8") as fh:
        for frame, tid, x, y, w, h, tail in all_rows:
            fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{tail}\n")
    return in_count, len(all_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear interpolation luk w plikach MOT.")
    p.add_argument("input", type=Path, help="Plik wejsciowy MOT lub katalog.")
    p.add_argument("output", type=Path, help="Plik/katalog wyjsciowy.")
    p.add_argument("--max-gap", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        for f in sorted(args.input.glob("*.txt")):
            n_in, n_out = interpolate_file(f, args.output / f.name, args.max_gap)
            print(f"  {f.name}: {n_in} -> {n_out} (+{n_out - n_in})")
    else:
        n_in, n_out = interpolate_file(args.input, args.output, args.max_gap)
        print(f"{args.input.name}: {n_in} -> {n_out} (+{n_out - n_in})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
