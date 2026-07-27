"""Text-to-speech: macOS `say`. v1-grade by design (see PLAN tradeoffs)."""

import asyncio


async def speak(text: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec("say", text)
        await proc.wait()
    except Exception as exc:
        print(f"[tts] say failed: {exc}", flush=True)
