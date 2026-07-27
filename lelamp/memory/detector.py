"""Object detection behind a narrow interface.

Backend is YOLOv8n (COCO-80, local, fast). The observer only sees
(label, score, bbox) tuples, so swapping backends — or adding vision-LLM
open-vocabulary labels later — touches nothing else.
"""

from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "yolov8n.pt"


class ObjectDetector:
    def __init__(self, model_path: Path = MODEL_PATH, min_score: float = 0.55):
        from ultralytics import YOLO  # deferred: ~1s import, observer-thread only

        self._model = YOLO(str(model_path))
        self.min_score = min_score

    def detect(self, frame_bgr) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        """Returns [(label, score, (cx, cy, w, h))] with normalized coords."""
        result = self._model.predict(frame_bgr, conf=self.min_score, verbose=False)[0]
        out = []
        for box in result.boxes:
            label = self._model.names[int(box.cls[0])]
            score = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxyn[0])
            out.append((label, score, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)))
        return out
