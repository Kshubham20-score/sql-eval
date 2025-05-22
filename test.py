from openai import OpenAI

client_openai = OpenAI(base_url="https://node4-api.staging.scorelabsai.com/v1", api_key="EMPTY")

messages= [
  {
    "role": "system",
    "content": "Your role is to convert a user question to a  query, given a database schema."
  },
  {
    "role": "user",
    "content": "Generate a SQL query that answers the question .\n\nThis query will run on a database whose schema is"
  }
]


response = client_openai.chat.completions.create(
                messages=messages,
                model="defog/llama-3-sqlcoder-8b",
                max_tokens=100,
                temperature=0.1,
                stop=[],
                # seed=0,
                # store=True,
                # metadata=None,
            )

print( response.choices[0].message.content)
