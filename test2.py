from openai import OpenAI
 
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://node4-api.staging.scorelabsai.com/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.chat.completions.create(model="defog/llama-3-sqlcoder-8b",
                                      messages=[{"role":"user", "content":"whats capital of france"}], max_tokens=100)
print("Completion result:", completion.choices[0].message.content)
