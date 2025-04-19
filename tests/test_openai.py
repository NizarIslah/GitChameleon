from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="bigcode/starcoder2-15b-instruct-v0.1",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message)
