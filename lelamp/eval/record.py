"""Record a labeled engagement clip. The script cues you (spoken + printed),
so ground-truth labels come from the schedule you were following.

  python -m lelamp.eval.record clips/plain --script engaged:5,away:5,engaged:3,away:5
  python -m lelamp.eval.record clips/glasses --script engaged:5,away:5   # wear glasses
  python -m lelamp.eval.record clips/absent --script engaged:3,absent:5,engaged:3

Labels: engaged (look at the screen), away (present, look well away),
absent (duck out of frame). Writes <name>.mp4 + <name>.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

CUES = {
    "engaged": "look at the lamp",
    "away": "look away now",
    "absent": "get out of frame",
}


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3 or args[1] != "--script":
        sys.exit(__doc__)
    out = Path(args[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    for part in args[2].split(","):
        label, _, dur = part.partition(":")
        if label not in CUES:
            sys.exit(f"unknown label {label!r} (use {'/'.join(CUES)})")
        segments.append((label, float(dur)))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("camera failed to open — check macOS camera permission")
    ok, frame = cap.read()
    if not ok:
        sys.exit("no frames from camera")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out.with_suffix(".mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    print("starting in 3s — follow the spoken cues")
    subprocess.Popen(["say", "get ready"])
    time.sleep(3)

    t0 = time.monotonic()
    timestamps: list[float] = []
    seg_times = []
    cursor = 0.0
    for label, dur in segments:
        seg_times.append({"label": label, "t0": cursor, "t1": cursor + dur})
        cursor += dur

    seg_idx = -1
    while True:
        t = time.monotonic() - t0
        if t >= cursor:
            break
        while seg_idx + 1 < len(seg_times) and t >= seg_times[seg_idx + 1]["t0"]:
            seg_idx += 1
            cue = CUES[seg_times[seg_idx]["label"]]
            print(f"[{t:5.1f}s] {cue}")
            subprocess.Popen(["say", cue])
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)
        timestamps.append(t)

    cap.release()
    writer.release()
    out.with_suffix(".json").write_text(json.dumps({
        "video": out.with_suffix(".mp4").name,
        "timestamps": timestamps,
        "segments": seg_times,
    }))
    subprocess.Popen(["say", "done"])
    print(f"saved {out.with_suffix('.mp4')} ({len(timestamps)} frames) + labels")


if __name__ == "__main__":
    main()
