"""MemoryStore dedupe test — the plan's policy: same label in same region
within the window = update, not insert.

Run: .venv/bin/python tests/test_store.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from lelamp.memory.store import MemoryStore

FRAME = np.full((480, 640, 3), 128, dtype=np.uint8)


def make_store(tmp):
    return MemoryStore(db_path=Path(tmp) / "test.db", thumb_dir=Path(tmp) / "thumbs")


def test_same_region_updates():
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        a = store.upsert("cup", (0.50, 0.50, 0.1, 0.1), 0.8, FRAME, now=1000.0)
        b = store.upsert("cup", (0.55, 0.52, 0.1, 0.1), 0.6, FRAME, now=1005.0)  # drifted a bit
        assert a == b, "same cup nearby should update, not insert"
        rows = store.query("cup")
        assert len(rows) == 1
        assert rows[0]["times_seen"] == 2
        assert rows[0]["score"] == 0.8, "score keeps the best detection"
        assert rows[0]["first_seen"] == 1000.0 and rows[0]["last_seen"] == 1005.0
        assert Path(rows[0]["thumbnail"]).exists()
        store.close()


def test_different_region_inserts():
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        a = store.upsert("cup", (0.2, 0.2, 0.1, 0.1), 0.8, FRAME, now=1000.0)
        b = store.upsert("cup", (0.8, 0.7, 0.1, 0.1), 0.8, FRAME, now=1005.0)
        assert a != b, "two cups far apart are two objects"
        assert len(store.query("cup")) == 2
        store.close()


def test_window_expiry_inserts():
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        a = store.upsert("cup", (0.5, 0.5, 0.1, 0.1), 0.8, FRAME, now=1000.0)
        b = store.upsert("cup", (0.5, 0.5, 0.1, 0.1), 0.8, FRAME, now=1000.0 + 700)  # beyond window
        assert a != b, "same spot much later is a fresh observation"
        store.close()


def test_moved_object_updates_position():
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        a = store.upsert("cell phone", (0.30, 0.40, 0.1, 0.1), 0.8, FRAME, now=1000.0)
        store.upsert("cell phone", (0.38, 0.45, 0.1, 0.1), 0.8, FRAME, now=1010.0)  # nudged
        row = store.query("phone")[0]
        assert row["id"] == a
        assert abs(row["cx"] - 0.38) < 1e-9, "record follows the object"
        store.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all store tests passed")
