# mcp_server.py
import json
import requests
from datetime import datetime
from dateutil import parser
from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Import helper to fetch tokens
from google_calendar_auth import get_access_token  # <-- you already wrote this in google_auth.py

mcp = FastMCP("GoogleCalendarMCP", log_level="ERROR")

CALENDAR_API = "https://www.googleapis.com/calendar/v3"
DEFAULT_CALENDAR_ID = "primary"


# ---------- Utility ----------

def list_google_events(access_token, time_min=None, time_max=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"singleEvents": "true", "orderBy": "startTime"}
    if time_min:
        params["timeMin"] = time_min
    if time_max:
        params["timeMax"] = time_max

    resp = requests.get(
        f"{CALENDAR_API}/calendars/{DEFAULT_CALENDAR_ID}/events",
        headers=headers,
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


def check_conflict(start_iso, end_iso, events):
    s = parser.isoparse(start_iso)
    e = parser.isoparse(end_iso)

    for ev in events.get("items", []):
        ev_s = parser.isoparse(ev["start"].get("dateTime", ev["start"].get("date")))
        ev_e = parser.isoparse(ev["end"].get("dateTime", ev["end"].get("date")))
        if not (e <= ev_s or s >= ev_e):  # overlap
            return True, ev
    return False, None


# ---------- MCP Tools ----------

@mcp.tool(
    name="list_events",
    description="List events from the user's Google Calendar within an optional time range (ISO 8601).",
)
def list_events(
    time_min: str | None = Field(
        None, description="Start of time range (ISO 8601). Example: 2025-09-06T00:00:00Z"
    ),
    time_max: str | None = Field(
        None, description="End of time range (ISO 8601). Example: 2025-09-06T23:59:59Z"
    ),
):
    access_token = get_access_token()
    return list_google_events(access_token, time_min, time_max)


@mcp.tool(
    name="create_event",
    description="Create a new calendar event, after checking for conflicts.",
)
def create_event(
    summary: str = Field(..., description="Title of the event"),
    start_iso: str = Field(..., description="Start datetime (ISO 8601, with timezone)"),
    end_iso: str = Field(..., description="End datetime (ISO 8601, with timezone)"),
    timezone: str = Field(..., description="Timezone string, e.g., Asia/Kolkata"),
    description: str | None = Field(None, description="Optional description"),
    attendees: list[str] | None = Field(None, description="List of attendee emails"),
):
    access_token = get_access_token()

    # Check conflicts
    events = list_google_events(access_token, start_iso, end_iso)
    conflict, ev = check_conflict(start_iso, end_iso, events)
    if conflict:
        return {"status": "conflict", "conflicting_event": ev}

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": timezone},
        "end": {"dateTime": end_iso, "timeZone": timezone},
    }
    if attendees:
        payload["attendees"] = [{"email": a} for a in attendees]

    resp = requests.post(
        f"{CALENDAR_API}/calendars/{DEFAULT_CALENDAR_ID}/events",
        headers=headers,
        data=json.dumps(payload),
    )
    resp.raise_for_status()
    return {"status": "created", "event": resp.json()}


@mcp.tool(
    name="update_event",
    description="Update an existing calendar event by ID. Also checks for conflicts if start/end changed.",
)
def update_event(
    event_id: str = Field(..., description="Google Calendar event ID"),
    start_iso: str | None = Field(None, description="Updated start time (ISO 8601)"),
    end_iso: str | None = Field(None, description="Updated end time (ISO 8601)"),
    timezone: str | None = Field(None, description="Timezone string"),
    summary: str | None = Field(None, description="Updated event title"),
    description: str | None = Field(None, description="Updated description"),
):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    # Fetch existing event
    current_event = requests.get(
        f"{CALENDAR_API}/calendars/{DEFAULT_CALENDAR_ID}/events/{event_id}",
        headers=headers,
    ).json()

    # If times updated, check conflicts
    if start_iso and end_iso:
        events = list_google_events(access_token, start_iso, end_iso)
        conflict, ev = check_conflict(start_iso, end_iso, events)
        if conflict and ev["id"] != event_id:
            return {"status": "conflict", "conflicting_event": ev}

    # Update payload
    if summary:
        current_event["summary"] = summary
    if description:
        current_event["description"] = description
    if start_iso and timezone:
        current_event["start"] = {"dateTime": start_iso, "timeZone": timezone}
    if end_iso and timezone:
        current_event["end"] = {"dateTime": end_iso, "timeZone": timezone}

    resp = requests.put(
        f"{CALENDAR_API}/calendars/{DEFAULT_CALENDAR_ID}/events/{event_id}",
        headers=headers,
        data=json.dumps(current_event),
    )
    resp.raise_for_status()
    return {"status": "updated", "event": resp.json()}


@mcp.tool(
    name="delete_event",
    description="Delete a calendar event by ID.",
)
def delete_event(
    event_id: str = Field(..., description="Google Calendar event ID to delete"),
):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    resp = requests.delete(
        f"{CALENDAR_API}/calendars/{DEFAULT_CALENDAR_ID}/events/{event_id}",
        headers=headers,
    )

    if resp.status_code == 204:
        return {"status": "deleted", "event_id": event_id}
    else:
        return {"status": "failed", "details": resp.text}


# ---------- MCP Resources ----------

@mcp.resource("cal://events", mime_type="application/json")
def all_events():
    """Expose all events as a resource (today onwards)."""
    access_token = get_access_token()
    now = datetime.utcnow().isoformat() + "Z"
    return list_google_events(access_token, now)


if __name__ == "__main__":
    mcp.run(transport="stdio")
