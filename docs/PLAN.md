# Build Plan — Expressive 6-DOF Lamp (Digital)

Challenge: a lamp that detects engagement, reacts expressively, remembers what it
sees, and answers questions about it via LLM. No hardware — the lamp is rendered
in 3D. All development on macOS (needs the camera).

## Core architecture

Four modules that only talk through two narrow interfaces. Perception never
touches the renderer; everything routes through the state machine.

```
camera ──> perception (attention @ 30fps) ──> engagement FSM ──> behavior engine ──> lamp viz (pose/light/sound)
   │                                                                  ▲
   └──> scene observer (objects @ ~1/5s) ──> memory store ────────────┤
                                                  ▲                   │
mic ──> STT ──> LLM (tool: query_memory) ──> TTS ─┴───────────────────┘
```

**Interface 1 — AttentionSignal.** Everything upstream of the FSM produces this;
the FSM consumes nothing else:

```python
@dataclass
class AttentionSignal:
    engaged: bool          # is the user attending to the lamp
    target: tuple | None   # normalized (x, y) of the user's face/attention
    confidence: float
    timestamp: float
```

Two interchangeable drivers: `CursorAttention` (dev scaffold — cursor hovering
on/near the lamp = engaged) and `CameraAttention` (MediaPipe head pose). The
swap is a one-line change; that's the point.

**Interface 2 — LampCommand.** The behavior engine emits pose targets, light
states, and sound cues; the renderer executes them with easing. The renderer
knows nothing about why.

## Stack

| Piece | Choice | Why |
|---|---|---|
| Core language | Python 3.11+ | MediaPipe/OpenCV/Whisper ecosystem |
| Lamp rendering | three.js in browser, vanilla JS, no build step | real 6-DOF articulation, best expressiveness per hour of work |
| Python ↔ viz bridge | FastAPI + WebSocket, JSON messages @ 30Hz | ~50 lines, clean module boundary |
| Face/gaze | MediaPipe Face Landmarker (head pose) | 30fps on CPU, free, reliable |
| Object detection | YOLOv8n locally; vision-LLM call for open-vocab labels | fast loop local, rich labels cloud |
| Memory store | SQLite: (label, region, timestamp, thumbnail path) | queryable, zero infra |
| STT | faster-whisper (small) | local, fast enough |
| TTS | macOS `say` for v1; upgrade later if time | zero setup |
| LLM | Claude API with tool use (`query_memory`) | native tool calling |

## Engagement FSM

States: `IDLE` (no face) → `ENGAGED` → `DISENGAGED` → `SEEKING` (attention-seeking).

Hysteresis everywhere, never instant transitions:
- ENGAGED after ~1.0s of sustained attention
- DISENGAGED after ~3.0s of sustained looking-away
- SEEKING after ~10s in DISENGAGED; back off after 2 failed attempts
- Signal smoothing: rolling window over raw AttentionSignal before the FSM sees it

Tune thresholds against the camera driver, not the cursor driver — cursor input
is noise-free and will lie about what works.

## Milestones

Each milestone has a verification gate. Don't start the next until it passes.

### M0 — Scaffolding
- Repo layout (below), interfaces defined, README.
- three.js lamp renders: base, 3 arm segments, head with emissive light cone.
- **Verify:** browser shows lamp; Python can push a pose over WebSocket and the lamp moves with easing.

### M1 — Behavior engine + cursor driver (demo steps 1–2, digital)
- `CursorAttention` driver (hover near lamp = engaged; elsewhere = disengaged).
- FSM with the four states; behavior library: greet, track target, droop on
  disengage, attention-seek (perk up, flash light, chirp).
- Ease-in/out on all motion (cubic or spring). Snappy linear motion reads as CNC, not creature.
- **Verify:** cursor hover → lamp greets and tracks; move away → droops; stay away → seeking behavior fires. Record a screen capture.

### M2 — Camera perception (Mac)
- `CameraAttention`: MediaPipe head pose → is the user facing the screen/lamp + face position.
- Smoothing + hysteresis; re-tune all FSM thresholds on real faces.
- Optional: add artificial jitter/dropout to CursorAttention to keep it honest as a test harness.
- **Verify:** demo steps 1–2 work with real gaze — glasses on, sitting at an angle, lights dimmed. No state flicker at threshold.

### M3 — Memory formation
- Scene observer: every ~5s (and on scene-change trigger), grab a frame, detect
  objects, upsert into SQLite with region + timestamp + thumbnail.
- Dedupe policy: same label in same region within N minutes = update, not insert.
- **Verify:** place keys/mug/phone in view, check DB rows have correct labels and sane regions; move an object, confirm the record updates.

### M4 — Conversation + recall (demo step 4)
- v1: text box in the viz page → LLM with `query_memory(label | description)` tool → text reply. Prove the memory path first.
- v2: push-to-talk → faster-whisper → LLM → `say`. Lamp does a "listening" pose during capture.
- **Verify:** "where are my keys?" answered correctly from the DB, including "when did you last see them?"

### M5 — Eval + writeup + video
- Engagement eval harness: record 10–15 short labeled clips (engaged/not per
  segment), replay through the pipeline, report precision/recall/F1 + flicker
  count. Include the hard cases: glasses, angle, low light.
- Latency: timestamp every pipeline stage; report p50/p95 for
  frame→engagement-decision, decision→motion-onset, and question→spoken-answer.
- Writeup: architecture diagram, data flow, tradeoffs (seed from the table below).
- Demo video following the 4-step scenario.

## Tradeoffs to argue in the writeup

These are graded — write them down as decisions are made, not retroactively.

- **Rule-based FSM vs learned engagement classifier** — FSM: debuggable,
  tunable, no training data needed; a classifier is the "touch weights" path and
  is overkill at this scale. Say so explicitly.
- **YOLO local vs vision-LLM for memory** — latency/cost vs open-vocabulary
  richness; hybrid chosen (YOLO for the loop, VLM for labels).
- **Cursor scaffold** — decoupled behavior development from perception noise;
  interface-first design made the swap trivial.
- **Web viz vs native/pygame** — real 6-DOF articulation and demo quality vs an
  extra process and a WebSocket. Fallback if the bridge fights back: 2D pygame lamp.
- **Local perception, cloud conversation** — 30fps loop must be local; the
  conversational path tolerates seconds of latency.

## Repo layout

```
lelamp/
  perception/      # AttentionSignal, cursor driver, camera driver
  behavior/        # FSM, behavior library, easing
  memory/          # scene observer, store, query API
  conversation/    # STT, LLM + tools, TTS
  viz/             # index.html + three.js lamp (static files)
  eval/            # engagement harness, latency probes
  server.py        # wires modules, serves viz, WebSocket
docs/
  PLAN.md          # this file
  ARCHITECTURE.md  # final writeup (M5)
```

## Risks

- **Threshold tuning eats the calendar** — start M2 early; M3/M4 are independent
  of M1/M2 and can proceed in parallel if blocked.
- **"Looking at the lamp" is ill-posed on a screen** — the lamp lives on the
  monitor, so "facing the screen + face detected" is the engagement proxy.
  State this assumption in the writeup rather than pretending to true gaze.
- **Whisper/TTS yak-shaving** — that's why v1 of conversation is a text box.

## Bonus (only if M0–M5 are done)

Interruption awareness is the cheapest standout: user is speaking (VAD on mic) →
suppress attention-seeking. It reuses the mic pipeline and shows social judgment.
