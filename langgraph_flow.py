# langgraph_flow.py
import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from mcp_client import MCPClient, sanitize_schema
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Gemini Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


# ---------- State Definition ----------
class AssistantState(TypedDict):
    query: str
    plan: str
    tool_name: str
    tool_args: dict
    tool_result: Any
    final_response: str


# ---------- Nodes ----------
async def planner_node(state: AssistantState) -> AssistantState:
    """Decide which tool to call based on the query."""
    client = MCPClient()
    await client.connect()
    tools = await client.list_tools()

    # Sanitize tool schemas
    tool_specs = []
    for t in tools:
        schema = sanitize_schema(dict(t.inputSchema))
        spec = {"name": t.name, "description": t.description, "parameters": schema}
        tool_specs.append(spec)

    SYSTEM_PROMPT = """
    You are a Google Calendar Assistant.
    Decide which tool to call based on the query.
    Only output JSON with { "tool_name": str, "tool_args": dict }.
    """

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[{"function_declarations": tool_specs}],
        system_instruction=SYSTEM_PROMPT,
    )

    chat = model.start_chat()
    response = chat.send_message(state["query"])

    fn_call = response.candidates[0].content.parts[0].function_call
    state["plan"] = f"Call {fn_call.name} with args {fn_call.args}"
    state["tool_name"] = fn_call.name
    state["tool_args"] = dict(fn_call.args)

    return state


async def executor_node(state: AssistantState) -> AssistantState:
    """Actually call the MCP tool."""
    client = MCPClient()
    await client.connect()

    result = await client.call_tool(state["tool_name"], state["tool_args"])
    state["tool_result"] = result
    return state


async def validator_node(state: AssistantState) -> AssistantState:
    """Validate tool result (basic error handling)."""
    if not state.get("tool_result"):
        state["final_response"] = "Sorry, I couldn't complete that action."
    else:
        state["final_response"] = f"Tool `{state['tool_name']}` executed successfully."
    return state


async def responder_node(state: AssistantState) -> AssistantState:
    """Generate a natural language response."""
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    chat = model.start_chat()

    follow_up = chat.send_message(
        f"User query: {state['query']}. "
        f"Tool result: {state['tool_result']}. "
        f"Now provide a natural, user-friendly response."
    )

    state["final_response"] = follow_up.text
    return state


# ---------- Graph ----------
def build_graph():
    workflow = StateGraph(AssistantState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "validator")
    workflow.add_edge("validator", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


# ---------- Run Helper ----------
async def run_langgraph(query: str) -> str:
    graph = build_graph()
    state = {"query": query}
    final_state = await graph.ainvoke(state)
    return final_state["final_response"]


if __name__ == "__main__":
    out = asyncio.run(run_langgraph("Create a meeting tomorrow at 10am"))
    print(out)
