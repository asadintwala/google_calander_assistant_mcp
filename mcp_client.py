# mcp_client.py
import asyncio
import json
from fastapi import FastAPI, Body
from pydantic import BaseModel, AnyUrl
from typing import Dict, Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Gemini Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


# ---------- FastAPI App ----------
app = FastAPI(title="Google Calendar Assistant")


# ---------- Request Schema ----------
class QueryRequest(BaseModel):
    query: str


# ---------- MCP Client Wrapper ----------
class MCPClient:
    def __init__(self, command: str = "python", args: list[str] = ["mcp_server.py"], env: Optional[dict] = None,):
        self._command = command
        self._args = args
        self._session: Optional[ClientSession] = None
        self.transport = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            # env=self._env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _stdio, _write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(_stdio, _write)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized or cache not populated. Call connect_to_server first."
            )
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(
        self, tool_name: str, tool_input
    ) -> types.CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(AnyUrl(uri))
        resource = result.contents[0]

        if isinstance(resource, types.TextResourceContents):
            if resource.mimeType == "application/json":
                return json.loads(resource.text)

            return resource.text

    async def cleanup(self):
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

def sanitize_schema(schema: dict) -> dict:
    """Remove unsupported JSON Schema fields for Gemini."""
    unsupported = {"title", "anyOf", "allOf", "oneOf", "not", "examples", "default"}
    clean = {}

    for k, v in schema.items():
        if k in unsupported:
            continue
        if isinstance(v, dict):
            clean[k] = sanitize_schema(v)
        elif isinstance(v, list):
            clean[k] = [sanitize_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            clean[k] = v
    return clean

def serialize_tool_result(result):
    """Convert CallToolResult or MCP content into JSON-serializable form."""
    if result is None:
        return None

    # If it's already a dict or str
    if isinstance(result, (dict, str, int, float, bool)):
        return result

    # If it's a list, serialize each element
    if isinstance(result, list):
        return [serialize_tool_result(r) for r in result]

    # If it has .to_dict() (MCP objects often do)
    if hasattr(result, "to_dict"):
        return result.to_dict()

    # If it's a Pydantic-like object
    if hasattr(result, "__dict__"):
        return {k: serialize_tool_result(v) for k, v in result.__dict__.items()}

    # Special handling for TextContent (common MCP return type)
    if hasattr(result, "text"):
        return result.text

    # Fallback: string representation
    return str(result)


# ---------- Gemini Helper ----------
async def run_with_gemini(user_query: str) -> str:
    """Send user query to Gemini with MCP tool schema and execute if tool call is returned."""

    async with MCPClient() as client:  # <-- FIXED: async context
        tools = await client.list_tools()

        # Convert tools to Gemini function-calling format
        tool_specs = []
        for t in tools:
            schema = sanitize_schema(dict(t.inputSchema))
            spec = {
                "name": t.name,
                "description": t.description,
                "parameters": schema,
            }
            tool_specs.append(spec)

        SYSTEM_PROMPT = """
        You are a Google Calendar Assistant.

        You have access to the following tools:
        1. list_events - to fetch events within a given time range
        2. create_event - to create new events in the calendar
        3. update_event - to update an existing event
        4. delete_event - to delete an event from the calendar

        Guidelines:
        - Always prefer using tools for answering queries instead of guessing.
        - When creating or updating an event, always check for conflicts first (do not allow overlapping events).
        - Respect timezones and ISO 8601 formats for datetime values.
        - When listing events, summarize them in a clear, human-friendly way.
        - If a user request is unclear, ask clarifying questions before taking action.
        - Never reveal raw tool inputs/outputs unless the user explicitly asks for JSON. 
        - Always respond in a natural conversational tone.

        Your job is to understand the user’s query, decide the correct tool call, execute it, and then summarize the results in a user-friendly response.
        """

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            tools=[{"function_declarations": tool_specs}],
            system_instruction=SYSTEM_PROMPT,
        )

        chat = model.start_chat()

        # Ask Gemini with tools
        response = chat.send_message(user_query)

        # Check if Gemini requested a tool call
        if response.candidates[0].content.parts[0].function_call:
            fn_call = response.candidates[0].content.parts[0].function_call
            tool_name = fn_call.name
            args = {k: v for k, v in fn_call.args.items()}

            # Call tool via MCP
            result = await client.call_tool(tool_name, args)
            tool_result = serialize_tool_result(result)  # <-- FIXED: serialization

            # Send result back to Gemini for final answer
            follow_up = chat.send_message(
                f"Tool `{tool_name}` executed. Result: {json.dumps(tool_result)}. "
                f"Now provide the final user-friendly response."
            )
            return follow_up.text

        return response.text


# ---------- FastAPI Endpoint ----------
@app.post("/assistant/query")
async def assistant_query(request: QueryRequest = Body(...)):
    final_answer = await run_with_gemini(request.query)
    return {"response": final_answer}
