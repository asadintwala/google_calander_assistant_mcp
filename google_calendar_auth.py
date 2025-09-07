# # google_auth.py
# import json
# import requests
# from pathlib import Path

# # File paths
# CREDENTIALS_FILE = Path("credentials.json")
# TOKEN_FILE = Path("token.json")

# # Load credentials.json (official Google structure)
# with open(CREDENTIALS_FILE, "r") as f:
#     creds = json.load(f)["installed"]

# CLIENT_ID = creds["client_id"]
# CLIENT_SECRET = creds["client_secret"]
# TOKEN_URI = creds["token_uri"]
# REDIRECT_URIS = creds["redirect_uris"]

# # Load token.json (contains refresh_token + scopes)
# with open(TOKEN_FILE, "r") as f:
#     token_data = json.load(f)

# REFRESH_TOKEN = token_data["refresh_token"]
# SCOPES = token_data.get("scopes", ["https://www.googleapis.com/auth/calendar"])


# def get_access_token() -> str:
#     """Use refresh token to get a fresh access token"""
#     data = {
#         "client_id": CLIENT_ID,
#         "client_secret": CLIENT_SECRET,
#         "refresh_token": REFRESH_TOKEN,
#         "grant_type": "refresh_token",
#     }

#     r = requests.post(TOKEN_URI, data=data)
#     r.raise_for_status()

#     response = r.json()
#     return response["access_token"]


# def check_calendar_connection():
#     """Verify Calendar API access with the current credentials"""
#     access_token = get_access_token()
#     headers = {"Authorization": f"Bearer {access_token}"}

#     resp = requests.get(
#         "https://www.googleapis.com/calendar/v3/users/me/calendarList",
#         headers=headers,
#     )

#     if resp.status_code == 200:
#         calendars = resp.json().get("items", [])
#         print("✅ Google OAuth connected successfully!")
#         print("Your calendars:")
#         for cal in calendars:
#             print(f"- {cal['summary']} (ID: {cal['id']})")
#     else:
#         print("❌ Failed to connect:", resp.status_code, resp.text)


# if __name__ == "__main__":
#     check_calendar_connection()

from google_auth_oauthlib.flow import InstalledAppFlow
import json

flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scopes=['https://www.googleapis.com/auth/calendar'])
credentials = flow.run_local_server(port=8080)
with open('token.json', 'w') as token:
    token.write(credentials.to_json())