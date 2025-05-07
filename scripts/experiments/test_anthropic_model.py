import requests
import os

API_KEY = os.getenv("ANTHROPIC_API_KEY", "REMOVED")  # noqa: E501
MODEL = os.getenv("MODEL", "claude-3-opus-20240229")

headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

data = {
    "model": MODEL,
    "max_tokens": 10,
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
}

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers=headers,
    json=data
)
print(f"Status code: {response.status_code}")
print(response.text)
