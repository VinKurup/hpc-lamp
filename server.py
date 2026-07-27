"""Wires the modules together: cursor events in, LampCommands out.

Pipeline per tick (30Hz): driver.poll -> smoother -> FSM -> behavior engine
-> broadcast. Swapping CursorAttention for CameraAttention (M2) is the one
line marked below.
"""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from lelamp.behavior import LampCommand
from lelamp.behavior.engine import BehaviorEngine
from lelamp.behavior.fsm import EngagementFSM
from lelamp.perception.cursor import CursorAttention
from lelamp.perception.smoothing import SignalSmoother

VIZ_DIR = Path(__file__).parent / "lelamp" / "viz"
TICK_HZ = 30

clients: set[WebSocket] = set()

# Load .env (KEY=VALUE lines) so the OpenRouter key doesn't need shell config.
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from lelamp.memory.store import MemoryStore

store = MemoryStore()  # chat queries work in both driver modes

USE_CAMERA = "--camera" in sys.argv
if USE_CAMERA:
    from lelamp.memory.observer import SceneObserver
    from lelamp.perception.camera import CameraAttention

    driver = CameraAttention()
    # Observer shares the attention driver's camera frames (one webcam, one capture).
    observer = SceneObserver(lambda: driver.latest_frame, store)
else:
    driver = CursorAttention()

chat = None
if os.environ.get("OPENROUTER_API_KEY"):
    from lelamp.conversation import Conversation

    chat = Conversation(store)

from lelamp.conversation.stt import Transcriber
from lelamp.conversation.tts import speak
from lelamp.eval.latency import LatencyProbes

stt = Transcriber()
probes = LatencyProbes()
smoother = SignalSmoother(window=0.5)
fsm = EngagementFSM()
engine = BehaviorEngine()


async def broadcast(cmd: LampCommand) -> None:
    data = json.dumps(cmd.to_message())
    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            clients.discard(ws)


async def handle_chat(ws: WebSocket, text: str, spoken: bool = False) -> None:
    if chat is None:
        reply = "I can't chat yet — put OPENROUTER_API_KEY=... in .env and restart me."
    else:
        try:
            reply = await chat.ask(text)
        except Exception as exc:
            reply = f"(chat error: {exc})"
    try:
        await ws.send_text(json.dumps({"type": "chat_reply", "text": reply}))
    except Exception:
        pass
    if spoken and not reply.startswith("("):
        asyncio.create_task(speak(reply))


async def handle_audio(ws: WebSocket, blob: bytes) -> None:
    t0 = time.monotonic()
    try:
        text = await asyncio.to_thread(stt.transcribe_bytes, blob)
    except Exception as exc:
        text = ""
        print(f"[stt] transcription failed: {exc}", flush=True)
    probes.record("stt", time.monotonic() - t0)
    if not text:
        try:
            await ws.send_text(json.dumps({"type": "chat_reply", "text": "(I didn't catch that)"}))
        except Exception:
            pass
        return
    try:
        await ws.send_text(json.dumps({"type": "transcript", "text": text}))
    except Exception:
        pass
    await handle_chat(ws, text, spoken=True)
    probes.record("question_to_answer_start", time.monotonic() - t0)


async def tick_loop() -> None:
    while True:
        now = time.monotonic()
        signal = smoother.push(driver.poll(now))
        state, transition = fsm.step(signal, now)
        if signal.confidence > 0:
            probes.record("frame_to_decision", now - signal.timestamp)
        if transition:
            print(f"[fsm] {transition[0].value} -> {transition[1].value}", flush=True)
        cmd = engine.step(state, transition, signal, now)
        if cmd is not None:
            await broadcast(cmd)
            if transition:
                # renderer adds <=1 frame (~16ms) + spring rise on top of this
                probes.record("decision_to_command_sent", time.monotonic() - now)
        await asyncio.sleep(1 / TICK_HZ)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_CAMERA:
        driver.start()
        observer.start()
        print(f"[camera] driver + scene observer running", flush=True)
    stt.preload()
    task = asyncio.create_task(tick_loop())
    yield
    task.cancel()
    if USE_CAMERA:
        observer.stop()
        driver.stop()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            data = await ws.receive()
            if data.get("type") == "websocket.disconnect":
                break
            if "bytes" in data and data["bytes"]:
                asyncio.create_task(handle_audio(ws, data["bytes"]))
                continue
            if not data.get("text"):
                continue
            msg = json.loads(data["text"])
            if msg.get("type") == "cursor" and not USE_CAMERA:
                driver.update(msg["x"], msg["y"], bool(msg["over_lamp"]), time.monotonic())
            elif msg.get("type") == "chat":
                asyncio.create_task(handle_chat(ws, msg.get("text", "")))
            elif msg.get("type") == "ptt":
                engine.set_listening(msg.get("state") == "start", fsm.state, time.monotonic())
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.get("/latency")
async def latency_report() -> dict:
    return probes.report()


app.mount("/", StaticFiles(directory=VIZ_DIR, html=True), name="viz")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
