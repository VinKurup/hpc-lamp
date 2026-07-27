"""Replay labeled clips through the real perception pipeline and grade it.

  python -m lelamp.eval.replay clips/*.json

Per clip and aggregate:
- precision/recall/F1 of the smoothed engagement signal vs scripted labels
  (frames within GRACE seconds of a segment boundary are excluded — the
  pipeline's hysteresis is *supposed* to lag there)
- raw signal toggle count vs expected transitions
- FSM ENGAGED<->other transition count vs expected (flicker if higher)

Uses the same landmarker, signal logic, smoother, and FSM as the live server.
"""

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from lelamp.behavior.fsm import EngagementFSM, State
from lelamp.perception.camera import make_landmarker, signal_from_result
from lelamp.perception.smoothing import SignalSmoother

GRACE = 1.5  # seconds around boundaries excluded from frame scoring


def grade_clip(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    cap = cv2.VideoCapture(str(meta_path.parent / meta["video"]))
    landmarker = make_landmarker()
    smoother = SignalSmoother(window=0.5)
    fsm = EngagementFSM()

    def label_at(t: float) -> str | None:
        for seg in meta["segments"]:
            if seg["t0"] <= t < seg["t1"]:
                return seg["label"]
        return None

    def near_boundary(t: float) -> bool:
        edges = [meta["segments"][0]["t0"]] + [s["t1"] for s in meta["segments"]]
        return any(abs(t - e) <= GRACE for e in edges)

    tp = fp = fn = tn = 0
    toggles = 0
    fsm_flips = 0
    prev_engaged = None
    prev_fsm = None
    for ts in meta["timestamps"]:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int(ts * 1000))
        smoothed = smoother.push(signal_from_result(result, ts))
        state, _ = fsm.step(smoothed, ts)

        if prev_engaged is not None and smoothed.engaged != prev_engaged:
            toggles += 1
        prev_engaged = smoothed.engaged
        fsm_engaged = state is State.ENGAGED
        if prev_fsm is not None and fsm_engaged != prev_fsm:
            fsm_flips += 1
        prev_fsm = fsm_engaged

        label = label_at(ts)
        if label is None or near_boundary(ts):
            continue
        truth = label == "engaged"
        if truth and smoothed.engaged:
            tp += 1
        elif truth:
            fn += 1
        elif smoothed.engaged:
            fp += 1
        else:
            tn += 1
    cap.release()

    expected_flips = sum(
        1 for a, b in zip(meta["segments"], meta["segments"][1:])
        if (a["label"] == "engaged") != (b["label"] == "engaged")
    )
    return {"clip": meta_path.stem, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "toggles": toggles, "fsm_flips": fsm_flips, "expected_flips": expected_flips}


def summarize(results: list[dict]) -> None:
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n{'clip':<16}{'frames':>7}{'P':>7}{'R':>7}{'F1':>7}{'toggles':>9}{'fsm':>5}{'expect':>8}")
    for r in results:
        scored = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        if scored == 0:
            print(f"{r['clip']:<16}{scored:>7}   -- no frames outside the grace window;"
                  f" use segments longer than {2 * GRACE:.0f}s --")
            continue
        p = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0
        rec = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0.0
        f = 2 * p * rec / (p + rec) if p + rec else 0.0
        print(f"{r['clip']:<16}{scored:>7}{p:>7.2f}{rec:>7.2f}{f:>7.2f}"
              f"{r['toggles']:>9}{r['fsm_flips']:>5}{r['expected_flips']:>8}")
    print(f"{'AGGREGATE':<16}{'':>7}{precision:>7.2f}{recall:>7.2f}{f1:>7.2f}")
    print("\nfsm > expect on any clip = flicker; toggles well above expect = noisy raw signal")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    summarize([grade_clip(p) for p in paths])


if __name__ == "__main__":
    main()
