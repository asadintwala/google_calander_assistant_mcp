# voice_agent.py
import asyncio
import os
from dotenv import load_dotenv
from langgraph_flow import run_langgraph
from livekit import api, rtc

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")


async def handle_voice_query(audio_input: str) -> str:
    """
    Simulate STT -> LangGraph -> TTS.
    Here `audio_input` is just text for simplicity.
    Replace with real STT (speech-to-text) later.
    """
    # 1. STT step (dummy passthrough for now)
    text_query = audio_input

    # 2. Run LangGraph workflow
    response_text = await run_langgraph(text_query)

    # 3. TTS step (dummy print for now, later send to LiveKit speaker)
    print(f"[VoiceAgent Response]: {response_text}")
    return response_text


async def main():
    # Connect to LiveKit room (skeleton, not full implementation yet)
    print("Connecting to LiveKit...")
    # Normally you'd use rtc.Room.connect here with LIVEKIT creds.
    # We'll just test with dummy query:
    await handle_voice_query("Create an event on Sept 10, 2025 from 3pm to 4pm for project sync")


if __name__ == "__main__":
    asyncio.run(main())
