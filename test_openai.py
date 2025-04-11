import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

query = "What is a stock?"
payload = {
    "inputs": f"<|system|>You are a helpful financial assistant.<|user|>{query}<|assistant|>"
}

try:
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            print("✅ Hugging Face Response:", result[0]["generated_text"])
        else:
            print("⚠️ No generated_text found in response.")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
except Exception as e:
    print("❌ Hugging Face Error:", e)
