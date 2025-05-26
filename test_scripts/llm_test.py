import requests
from dotenv import load_dotenv
import os
load_dotenv()

# Replace with your actual OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Sample prompt to test
prompt = "Summarize this text: The video is amazing. I loved the editing and transitions. Very informative too! I believe that watching this video is the key to success in studying for final in the last minute."

# Headers required by OpenRouter
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "test-script",  # any name is fine
    "Content-Type": "application/json"
}

# API body using a model (you can change to gpt-3.5 or mistral, etc.)
body = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

# Make the API call
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=body
)

# Print result
if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print("✅ Response from LLM:\n")
    print(reply)
else:
    print(f"❌ Error {response.status_code}: {response.text}")

