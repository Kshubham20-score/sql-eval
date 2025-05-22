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


from utils.llm import chat_openai
import sqlparse

response = chat_openai(messages=messages, model="defog/llama-3-sqlcoder-8b", temperature=0.0, base_url="https://node4-api.staging.scorelabsai.com/v1")
generated_query = (
    response.content.split("```sql", 1)[-1].split("```", 1)[0].strip()
)
try:
    generated_query = sqlparse.format(
        generated_query, reindent=True, keyword_case="upper"
    )

except:
    pass

print("generated query:",generated_query)
