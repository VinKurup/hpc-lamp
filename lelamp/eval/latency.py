"""In-process latency collection: named stages, rolling window, percentiles."""

from collections import defaultdict, deque


class LatencyProbes:
    def __init__(self, window: int = 2000):
        self._samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def record(self, stage: str, seconds: float) -> None:
        self._samples[stage].append(seconds)

    def report(self) -> dict:
        out = {}
        for stage, samples in self._samples.items():
            data = sorted(samples)
            n = len(data)
            if not n:
                continue
            out[stage] = {
                "count": n,
                "p50_ms": round(data[n // 2] * 1000, 1),
                "p95_ms": round(data[min(n - 1, int(n * 0.95))] * 1000, 1),
                "max_ms": round(data[-1] * 1000, 1),
            }
        return out
