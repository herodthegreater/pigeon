"""pigeon -- detect and track pigeons with a YOLO detector.

First-light uses COCO's 'bird' (class 14) as a stand-in for 'pigeon' so we
have something working today with zero training data. Later we fine-tune on
real pigeons (so it ignores crows and gulls) and carve the model down with
the harness.

    python pigeon.py --source samples/pigeons.jpg --save
    python pigeon.py --source video.mp4 --track --save
    python pigeon.py --source 0 --track          # webcam
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

BIRD_CLASS = 14  # COCO index for 'bird' -- our proxy for 'pigeon'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--source", default="0",
                    help="image/video path, directory, URL, or webcam index")
    ap.add_argument("--track", action="store_true",
                    help="track birds across frames (for video/webcam)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true",
                    help="write annotated output under runs/")
    ap.add_argument("--device", default=None, help="mps, cpu (default: auto)")
    args = ap.parse_args()

    model = YOLO(args.model)
    kw = dict(
        source=args.source,
        classes=[BIRD_CLASS],       # <- the whole 'only pigeons' trick, for now
        conf=args.conf,
        save=args.save,
        project="runs",
        name="pigeon",
        exist_ok=True,
        verbose=False,
    )
    if args.device:
        kw["device"] = args.device

    runner = model.track(persist=True, **kw) if args.track else model.predict(**kw)

    total = 0
    for r in runner:
        total += 0 if r.boxes is None else len(r.boxes)
    print(f"\npigeon: {total} bird-detection(s) across the source "
          f"(class={BIRD_CLASS}, model={args.model}).")


if __name__ == "__main__":
    main()
