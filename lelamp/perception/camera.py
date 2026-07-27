"""CameraAttention — MediaPipe head-pose driver.

"Attending to the lamp" is proxied as: a face is detected AND it's roughly
facing the screen (the lamp lives on the monitor, so true gaze is ill-posed;
this assumption goes in the writeup). Head pose comes from the landmarker's
facial transformation matrix.

A background thread runs the camera at ~30fps and stores the latest signal;
poll() is non-blocking, same contract as CursorAttention.

Standalone check:  python -m lelamp.perception.camera [--preview]
"""

import math
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .attention import AttentionSignal

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

# Facing-the-screen thresholds, degrees. Tune against real faces (M2 gate).
YAW_LIMIT = 25.0
PITCH_MIN = -30.0   # looking down a bit is fine (lamp is on the desk/screen)
PITCH_MAX = 20.0

MIRROR = True  # selfie-view x so moving right moves the target right


def _pose_angles(matrix) -> tuple[float, float]:
    """(yaw, pitch) in degrees from a 4x4 facial transformation matrix."""
    # R = matrix[:3,:3]; standard ZYX decomposition, we need yaw (Y) and pitch (X).
    r = matrix
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2][0]))))
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    return yaw, pitch


def signal_from_result(result, now: float) -> AttentionSignal:
    """Landmarker result -> AttentionSignal. Shared by the live driver and the
    offline eval replay so both grade the exact same logic."""
    if not result.face_landmarks:
        return AttentionSignal(engaged=False, target=None, confidence=0.0, timestamp=now)
    # Face center from the landmark bounding box (normalized coords).
    xs = [lm.x for lm in result.face_landmarks[0]]
    ys = [lm.y for lm in result.face_landmarks[0]]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    if MIRROR:
        cx = 1.0 - cx
    facing = False
    if result.facial_transformation_matrixes:
        yaw, pitch = _pose_angles(result.facial_transformation_matrixes[0])
        facing = abs(yaw) <= YAW_LIMIT and PITCH_MIN <= pitch <= PITCH_MAX
    return AttentionSignal(
        engaged=facing,
        target=(cx, cy),
        confidence=1.0,
        timestamp=now,
    )


def make_landmarker():
    """A fresh FaceLandmarker in VIDEO mode (live driver and replay both use this)."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


class CameraAttention:
    def __init__(self, camera_index: int = 0, stale_after: float = 0.5):
        self.stale_after = stale_after
        self._camera_index = camera_index
        self._lock = threading.Lock()
        self._latest: AttentionSignal | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps = 0.0  # measured, for the latency report
        self.latest_frame = None  # BGR, debug/preview only

    def start(self) -> None:
        self._landmarker = make_landmarker()
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"camera {self._camera_index} failed to open — check macOS camera permission"
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def _loop(self) -> None:
        frames = 0
        window_start = time.monotonic()
        while self._running:
            ok, frame = self._cap.read()
            now = time.monotonic()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, int(now * 1000))
            sig = self._to_signal(result, frame.shape, now)
            with self._lock:
                self._latest = sig
                self.latest_frame = frame
            frames += 1
            if now - window_start >= 2.0:
                self.fps = frames / (now - window_start)
                frames = 0
                window_start = now

    def _to_signal(self, result, shape, now: float) -> AttentionSignal:
        return signal_from_result(result, now)

    def poll(self, now: float) -> AttentionSignal:
        with self._lock:
            sig = self._latest
        if sig is None or now - sig.timestamp > self.stale_after:
            return AttentionSignal(engaged=False, target=None, confidence=0.0, timestamp=now)
        return sig


if __name__ == "__main__":
    import sys

    driver = CameraAttention()
    driver.start()
    preview = "--preview" in sys.argv
    print("camera running; ctrl-c to stop")
    try:
        while True:
            time.sleep(0.5)
            sig = driver.poll(time.monotonic())
            print(
                f"fps={driver.fps:5.1f}  engaged={sig.engaged!s:5}  "
                f"target={sig.target}  conf={sig.confidence:.1f}"
            )
            if preview and driver.latest_frame is not None:
                frame = driver.latest_frame.copy()
                label = "ENGAGED" if sig.engaged else ("face" if sig.confidence else "no face")
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0) if sig.engaged else (0, 0, 255), 2)
                cv2.imshow("camera", frame)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        driver.stop()
