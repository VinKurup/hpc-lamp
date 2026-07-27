"""Conversation: LLM with a query_memory tool over the lamp's SQLite memory.

Routed through OpenRouter (OpenAI-compatible API) to a Claude model; the
model is env-overridable. The tool returns pre-humanized sightings — relative
times and user-frame positions are computed here deterministically, so the
LLM only composes the answer, never does coordinate math.
"""

import json
import os
import time

from openai import AsyncOpenAI

MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")

SYSTEM = """You are LeLamp, a small expressive desk lamp with a camera, sitting on the user's desk.
You remember objects you have seen. Use the query_memory tool to look things up before answering
questions about objects, locations, or when things were last seen.
Answer in one or two short, friendly spoken sentences. Include when you last saw the object and
where. Positions from the tool are already in the user's frame of reference — repeat them as given.
If memory has no matching object, say you haven't seen it."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Search the lamp's memory of objects seen by its camera, most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Object name substring, e.g. 'cup', 'phone'. Omit to list everything remembered.",
                    }
                },
                "required": [],
            },
        },
    }
]


def describe_sighting(row: dict, now: float) -> dict:
    age = max(0.0, now - row["last_seen"])
    if age < 90:
        when = f"{int(age)} seconds ago"
    elif age < 5400:
        when = f"{int(age / 60)} minutes ago"
    else:
        when = f"{age / 3600:.1f} hours ago"
    # Camera frames are unmirrored: image-left is the user's right.
    cx = row["cx"]
    if cx < 0.42:
        side = "to your right"
    elif cx > 0.58:
        side = "to your left"
    else:
        side = "right in front of you"
    return {
        "object": row["label"],
        "last_seen": when,
        "position": side,
        "times_seen": row["times_seen"],
    }


class Conversation:
    def __init__(self, store):
        self.store = store
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=30,  # SDK default is 600s — a hang must fail fast, not sit on "…"
            max_retries=1,
        )

    async def ask(self, text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ]
        for _ in range(4):  # tool-call round limit
            resp = await self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, max_tokens=400
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "(no answer)"
            messages.append(msg.model_dump(exclude_none=True))
            now = time.time()
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                # min_seen=2: single sightings are usually detector ghosts.
                rows = self.store.query(args.get("label"), min_seen=2, limit=10)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps([describe_sighting(r, now) for r in rows]),
                })
        return "Sorry — I got lost thinking about that one."
