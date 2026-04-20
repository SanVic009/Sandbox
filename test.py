import anthropic

def test_api_key(api_key: str):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("✅ API key is valid!")
        print(f"Response: {response.content[0].text}")
    except anthropic.AuthenticationError:
        print("❌ Invalid API key.")
    except anthropic.PermissionDeniedError:
        print("❌ Key exists but lacks permissions.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")

test_api_key("YOUR_API_KEY_HERE")
