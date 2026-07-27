# LeLamp Challenge — Expressive Digital Lamp

A 6-DOF lamp (rendered in 3D) that detects when you're paying attention, reacts
with expressive motion/light/sound, remembers objects it sees, and answers
questions about them via LLM.

Built for the Human Computer Lab SW/ML intern challenge.

## Status

M0–M4 complete (viz, cursor scaffold, camera engagement, memory, text+voice
recall). M5 in progress: eval clips + demo video. See
[docs/PLAN.md](docs/PLAN.md) for milestones and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design writeup.

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
pip install -r requirements.txt
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # for chat/voice recall
python server.py --camera                    # full stack, viz at localhost:8000
python server.py                             # cursor-scaffold mode (no camera)
```

Useful extras:

```
python tests/test_fsm.py && python tests/test_store.py       # unit tests
python -m lelamp.perception.camera --preview                 # tune engagement thresholds
python -m lelamp.eval.record clips/plain --script engaged:5,away:5,engaged:5,away:5
python -m lelamp.eval.replay clips/*.json                    # P/R/F1 + flicker
curl localhost:8000/latency                                  # stage latency percentiles
```
