# LeLamp Challenge — Expressive Digital Lamp

A 6-DOF lamp (rendered in 3D) that detects when you're paying attention, reacts
with expressive motion/light/sound, remembers objects it sees, and answers
questions about them via LLM.

Built for the Human Computer Lab SW/ML intern challenge.

## Status

Planning. See [docs/PLAN.md](docs/PLAN.md) for architecture, milestones, and
tradeoffs.

## Quick architecture

```
camera ──> perception (attention @ 30fps) ──> engagement FSM ──> behavior engine ──> lamp viz
   │                                                                  ▲
   └──> scene observer (objects) ──> memory store ────────────────────┤
                                          ▲                           │
mic ──> STT ──> LLM (query_memory) ──> TTS┴───────────────────────────┘
```

Python core (MediaPipe, YOLO, Whisper, Claude API) + three.js lamp renderer in
the browser, bridged over a WebSocket.

## Dev setup

Developed on macOS (camera required for M2+).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # coming with M0
python server.py                  # serves viz at localhost:8000
```
