import json
import requests

# تنظیمات اتصال
API_KEY = "sk-EQJTTrDxlUL3mdlW1HvdmDFRkBDOFUkfbjW5RsdJwZOyWTp0"  # کلید API خود را وارد کنید
URL = "https://agentrouter.org/v1/chat/completions"

# هدرهای سفارشی برای شبیه‌سازی درخواست Claude Code / Anthropic CLI
headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json",
    # شبیه‌سازی User-Agent مربوط به کلاد کد و درخواست‌های رسمی
    "User-Agent": "claude-code/0.1.0 (x86_64-apple-darwin; node/v20.10.0)",
    "X-Client-Name": "claude-code",
    "Accept": "application/json",
}

# بدنه درخواست (Payload)
payload = {
    "model": "claude-opus-5",
    "messages": [
        {
            "role": "user",
            "content": "Hello! Reply in one short sentence if you read this.",
        }
    ],
    "temperature": 0.7,
}

try:
    print("Sending request disguised as Claude Code...")
    response = requests.post(URL, headers=headers, data=json.dumps(payload))

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n--- Response ---")
        print(result["choices"][0]["message"]["content"])
    else:
        print("\n--- Error Response ---")
        print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")