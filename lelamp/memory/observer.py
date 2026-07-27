"""Scene observer: periodically detect objects and remember them.

Runs in its own thread off a shared frame source (the attention driver's
camera — two captures can't open one webcam). Detects every `interval`
seconds, or immediately when the scene changes (cheap downscaled frame diff).
People are not "objects the lamp remembers"; the person is handled by the
attention pipeline.

Standalone check (owns the camera):  python -m lelamp.memory.observer [secs]
"""

import threading
import time

import cv2

from .detector import ObjectDetector
from .store import MemoryStore

SCENE_DIFF_THRESHOLD = 14.0  # mean abs gray diff (0-255) on a 64x48 downscale


class SceneObserver:
    def __init__(
        self,
        get_frame,                 # () -> BGR frame | None
        store: MemoryStore,
        interval: float = 5.0,
        ignore_labels: tuple = ("person",),
    ):
        self.get_frame = get_frame
        self.store = store
        self.interval = interval
        self.ignore_labels = ignore_labels
        self._running = False
        self._thread: threading.Thread | None = None
        self._prev_small = None
        self.last_detections: list = []  # debug/verify

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _scene_changed(self, frame) -> bool:
        small = cv2.cvtColor(cv2.resize(frame, (64, 48)), cv2.COLOR_BGR2GRAY)
        changed = False
        if self._prev_small is not None:
            diff = cv2.absdiff(small, self._prev_small).mean()
            changed = diff > SCENE_DIFF_THRESHOLD
        self._prev_small = small
        return changed

    CHANGE_MIN_GAP = 3.0  # a moving person trips the diff constantly; rate-limit it

    def _loop(self) -> None:
        detector = ObjectDetector()  # load model in-thread (~1s)
        last_detect = 0.0
        while self._running:
            time.sleep(1.0)
            frame = self.get_frame()
            if frame is None:
                continue
            now = time.monotonic()
            due = now - last_detect >= self.interval
            changed = self._scene_changed(frame) and now - last_detect >= self.CHANGE_MIN_GAP
            if not due and not changed:
                continue
            last_detect = now
            detections = detector.detect(frame)
            self.last_detections = detections
            for label, score, bbox in detections:
                if label in self.ignore_labels:
                    continue
                self.store.upsert(label, bbox, score, frame_bgr=frame)


if __name__ == "__main__":
    import sys

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("camera failed to open — check macOS camera permission")
    latest = {"frame": None}

    def pump():
        while True:
            ok, frame = cap.read()
            if ok:
                latest["frame"] = frame

    threading.Thread(target=pump, daemon=True).start()
    store = MemoryStore()
    obs = SceneObserver(lambda: latest["frame"], store)
    obs.start()
    print(f"observing for {seconds:.0f}s ...")
    time.sleep(seconds)
    obs.stop()
    cap.release()
    for row in store.query():
        print(
            f"#{row['id']:<3} {row['label']:<14} seen x{row['times_seen']:<3}"
            f" center=({row['cx']:.2f},{row['cy']:.2f})"
            f" last={time.strftime('%H:%M:%S', time.localtime(row['last_seen']))}"
            f" thumb={row['thumbnail']}"
        )
