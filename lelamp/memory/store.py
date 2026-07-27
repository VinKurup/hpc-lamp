"""SQLite memory store for observed objects.

Dedupe policy (the plan's): the same label whose center lands within
DEDUPE_RADIUS of an existing record seen in the last DEDUPE_WINDOW is the
same object — update it (position, last_seen, times_seen, thumbnail) instead
of inserting. Wall-clock (time.time) timestamps so "when did you last see my
mug" has a human answer in M4.
"""

import sqlite3
import threading
import time
from pathlib import Path

import cv2

DEDUPE_RADIUS = 0.15   # normalized center distance
DEDUPE_WINDOW = 600.0  # seconds

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    cx REAL NOT NULL, cy REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
    score REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    times_seen INTEGER NOT NULL DEFAULT 1,
    thumbnail TEXT
)
"""


class MemoryStore:
    def __init__(self, db_path: str | Path = "memory.db", thumb_dir: str | Path = "thumbnails"):
        self.thumb_dir = Path(thumb_dir)
        self.thumb_dir.mkdir(exist_ok=True)
        # Observer thread writes, server/M4 queries read: one connection + lock.
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(SCHEMA)
            self._db.commit()

    def upsert(self, label: str, bbox: tuple, score: float, frame_bgr=None, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cx, cy, w, h = bbox
        with self._lock:
            rows = self._db.execute(
                "SELECT id, cx, cy FROM observations WHERE label = ? AND last_seen >= ?",
                (label, now - DEDUPE_WINDOW),
            ).fetchall()
            match = None
            best = DEDUPE_RADIUS
            for row in rows:
                dist = ((row["cx"] - cx) ** 2 + (row["cy"] - cy) ** 2) ** 0.5
                if dist <= best:
                    match, best = row["id"], dist
            if match is not None:
                self._db.execute(
                    "UPDATE observations SET cx=?, cy=?, w=?, h=?, score=MAX(score, ?),"
                    " last_seen=?, times_seen=times_seen+1 WHERE id=?",
                    (cx, cy, w, h, score, now, match),
                )
                obs_id = match
            else:
                cur = self._db.execute(
                    "INSERT INTO observations (label, cx, cy, w, h, score, first_seen, last_seen)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (label, cx, cy, w, h, score, now, now),
                )
                obs_id = cur.lastrowid
            if frame_bgr is not None:
                thumb = self._save_thumbnail(obs_id, frame_bgr, bbox)
                self._db.execute("UPDATE observations SET thumbnail=? WHERE id=?", (thumb, obs_id))
            self._db.commit()
        return obs_id

    def _save_thumbnail(self, obs_id: int, frame_bgr, bbox: tuple) -> str:
        fh, fw = frame_bgr.shape[:2]
        cx, cy, w, h = bbox
        x1 = max(0, int((cx - w / 2) * fw))
        y1 = max(0, int((cy - h / 2) * fh))
        x2 = min(fw, int((cx + w / 2) * fw))
        y2 = min(fh, int((cy + h / 2) * fh))
        path = self.thumb_dir / f"obs_{obs_id}.jpg"
        if x2 > x1 and y2 > y1:
            cv2.imwrite(str(path), frame_bgr[y1:y2, x1:x2])
        return str(path)

    def query(self, label: str | None = None, min_seen: int = 1, limit: int = 50) -> list[dict]:
        """Substring label match, most recently seen first. M4's tool calls this
        with min_seen=2: single-sighting records are usually detector ghosts, so
        they stay stored (storage is honest) but out of answers."""
        sql = "SELECT * FROM observations WHERE times_seen >= ?"
        args: list = [min_seen]
        if label:
            sql += " AND label LIKE ?"
            args.append(f"%{label}%")
        sql += " ORDER BY last_seen DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._db.close()
