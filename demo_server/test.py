#!/usr/bin/env python3
import asyncio
import json
import uuid

import websockets
import google.generativeai as genai
from google.generativeai import types

# ——— Configuration ———
GENAI_API_KEY = "YOUR_GOOGLE_API_KEY"         # ← replace with your key
MCP_WS_URL     = "ws://localhost:8765"        # ← your MCP server URL
MODEL_NAME     = "gemini-2.0-flash"

# ——— Initialize Gemini ———
genai.configure(api_key=GENAI_API_KEY)


async def main():
    # 1) Connect to MCP WebSocket
    print("⏳ Connecting to MCP server…")
    ws = await websockets.connect(MCP_WS_URL)

    # 2) Receive initial 'connected' event + full tool list
    greeting = json.loads(await ws.recv())
    tool_defs = greeting.get("tools", [])
    print("✅ Connected!")
    print("Available tools:")
    for t in tool_defs:
        print(f"  • {t['name']}: {t['description']}")

    # Wrap into a single Gemni function-calling object
    gemini_tools = [ types.Tool(function_declarations=tool_defs) ]

    # 3) Interactive chat loop
    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break

        # 3a) Ask Gemini which tool to call
        selection_prompt = f"""
You are an assistant that has exactly these tools available:
{json.dumps(tool_defs, indent=2)}

User wants: "{user_text}"

Pick exactly one tool to call, and output ONLY a JSON object:
{{"name":"<tool_name>","arguments":{{ ... }}}}
"""
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=genai.GenerationConfig(temperature=0.0)
        )

        response = model.generate_content(selection_prompt, tools=gemini_tools)
        candidate = response.candidates[0].content.parts[0]
        fc = candidate.function_call
        if not fc:
            print("⚠️  Gemini did not pick a tool, got:", candidate.text)
            continue

        tool_name = fc.name
        try:
            tool_args = json.loads(fc.args or "{}")
        except json.JSONDecodeError:
            print("⚠️  Couldn’t parse function_call.args:", fc.args)
            continue

        print(f"→ Calling `{tool_name}` with args {tool_args}")

        # 3b) Send it to MCP server
        req_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            "action":     "call_tool",
            "request_id": req_id,
            "tool_name":  tool_name,
            "args":       tool_args
        }))

        # 3c) Await the matching tool_result
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if (msg.get("event") == "tool_result"
             and msg.get("request_id") == req_id):
                result = msg["result"]
                print(f"✅ Result from `{tool_name}`:\n{json.dumps(result, indent=2)}")
                break

    # 4) Clean up
    await ws.close()
    print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
