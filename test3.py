import time
import sqlparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from concurrent.futures import ThreadPoolExecutor, as_completed

# Define costs (minimal set for compatibility)
LLM_COSTS_PER_TOKEN = {
    "Qwen/Qwen2.5-Coder-14B": {"input_cost_per1k": 0.0001, "output_cost_per1k": 0.0005},  # Example costs
}

@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    output_tokens_details: Optional[Dict[str, int]] = None
    cost_in_cents: Optional[float] = None

    def __post_init__(self):
        model_name = None
        for mname in LLM_COSTS_PER_TOKEN:
            if mname in self.model:
                model_name = mname
                break
        if model_name:
            self.cost_in_cents = (
                (self.input_tokens / 1000 * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]) +
                (self.output_tokens / 1000 * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"])
            ) * 100

def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "Qwen/Qwen2.5-Coder-14B",
    max_completion_tokens: int = 100,
    temperature: float = 0.1,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
    base_url: str = None,
) -> LLMResponse:
    from openai import OpenAI

    if base_url:
        client_openai = OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        client_openai = OpenAI()

    t = time.time()
    response = client_openai.chat.completions.create(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        store=store,
        metadata=metadata,
        timeout=timeout,
    )

    if len(response.choices) == 0:
        raise Exception("No response")

    content = response.choices[0].message.content
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        output_tokens_details=response.usage.completion_tokens_details,
    )

# Define the input messages (replace with your real prompt)
#messages = [
#    {"role": "system", "content": "You are a helpful assistant that writes SQL queries."},
#    {"role": "user", "content": "What is the average number of citations received by publications in each year?"}
#]

messages = [{'role': 'system', 'content': 'Your role is to convert a user question to a postgres query, given a database schema.'}, {'role': 'user', 'content': "Generate a SQL query that answers the question `What is the average number of citations received by publications in each year?`.\n\nThis query will run on a database whose schema is represented in this string:\nCREATE TABLE cite (\n  cited bigint, --ID of the publication being cited\n  citing bigint --ID of the publication that is citing another publication\n);\nCREATE TABLE author (\n  aid bigint, --Unique identifier for each author\n  oid bigint, --Foreign key referencing the organization the author belongs to\n  homepage text, --URL of the author's personal website\n  name text --Name of the author\n);\nCREATE TABLE domain (\n  did bigint, --Unique identifier for a domain\n  name text --Name of the domain\n);\nCREATE TABLE writes (\n  aid bigint, --Foreign key referencing the author table's primary key\n  pid bigint --Foreign key referencing the publication table's primary key\n);\nCREATE TABLE journal (\n  jid bigint, --Unique identifier for a journal\n  homepage text, --The homepage URL for the journal\n  name text --The name of the journal\n);\nCREATE TABLE keyword (\n  kid bigint, --Unique identifier for a keyword\n  keyword text --The actual keyword\n);\nCREATE TABLE conference (\n  cid bigint, --Unique identifier for a conference\n  homepage text, --The homepage URL for the conference\n  name text --The name of the conference\n);\nCREATE TABLE publication (\n  year bigint, --The year of publication\n  cid bigint, --The ID of the conference where the publication was presented\n  citation_num bigint, --The number of citations received by the publication\n  jid bigint, --The ID of the journal where the publication was published\n  pid bigint, --The unique ID of the publication\n  reference_num bigint, --The number of references cited by the publication\n  title text, --The title of the publication\n  abstract text --The abstract of the publication\n);\nCREATE TABLE organization (\n  oid bigint, --Unique identifier for the organization\n  continent text, --Continent where the organization is located\n  homepage text, --URL of the organization's homepage\n  name text --Name of the organization\n);\nCREATE TABLE domain_author (\n  aid bigint, --Foreign key referencing the author table's primary key\n  did bigint --Foreign key referencing the domain table's primary key\n);\nCREATE TABLE domain_journal (\n  did bigint, --Foreign key referencing the domain table's primary key\n  jid bigint --Foreign key referencing the journal table's primary key\n);\nCREATE TABLE domain_keyword (\n  did bigint, --Foreign key referencing the 'did' column of the 'domain' table\n  kid bigint --Foreign key referencing the 'kid' column of the 'keyword' table\n);\nCREATE TABLE domain_conference (\n  cid bigint, --Foreign key referencing the cid column in the conference table\n  did bigint --Foreign key referencing the did column in the domain table\n);\nCREATE TABLE domain_publication (\n  did bigint, --Foreign key referencing the domain table's primary key column (did)\n  pid bigint --Foreign key referencing the publication table's primary key column (pid)\n);\nCREATE TABLE publication_keyword (\n  pid bigint, --Foreign key referencing the publication table's primary key (pid)\n  kid bigint --Foreign key referencing the keyword table's primary key (kid)\n);\n\nHere is a list of joinable columns:\nauthor.aid can be joined with domain_author.aid\nauthor.oid can be joined with organization.oid\nauthor.aid can be joined with writes.aid\ncite.cited can be joined with publication.pid\ncite.citing can be joined with publication.pid\nconference.cid can be joined with domain_conference.cid\nconference.cid can be joined with publication.cid\ndomain.did can be joined with domain_author.did\ndomain.did can be joined with domain_conference.did\ndomain.did can be joined with domain_journal.did\ndomain.did can be joined with domain_keyword.did\ndomain_journal.jid can be joined with journal.jid\ndomain_keyword.kid can be joined with keyword.kid\ndomain_publication.pid can be joined with publication.pid\njournal.jid can be joined with publication.jid\nkeyword.kid can be joined with publication_keyword.kid\npublication.pid can be joined with publication_keyword.pid\npublication.pid can be joined with writes.pid\n\nReturn only the SQL query, and nothing else."}]


# Run 10 sequential requests
#for i in range(10):
def make_request(i):
    print(f"\n=== Request {i+1} started  ===")
    try:
        response = chat_openai(
            messages=messages,
            model="Qwen/Qwen2.5-Coder-14B",
            temperature=0.1,
            base_url="https://node4-api.staging.greenjello.io/v1",  # <-- replace with your actual base URL
        )

        content = response.content
        print("[RAW OUTPUT]:", content)

        # Try to extract SQL from markdown-style block
        try:
            generated_query = content.split("```sql", 1)[-1].split("```", 1)[0].strip()
        except IndexError:
            generated_query = content.strip()

        # Format SQL
        try:
            generated_query = sqlparse.format(generated_query, reindent=True, keyword_case="upper")
        except Exception:
            pass

        print("[FORMATTED SQL]:", generated_query)
    except Exception as e:
        print(f"[ERROR]: {e}")

total_requests =10
max_parallel = 4

with  ThreadPoolExecutor(max_workers=max_parallel) as executor:
    futures = [executor.submit(make_request,i) for i in range(total_requests)]
    for future in as_completed(futures, timeout=300):
        result = future.result(timeout=90)
        if result is not None:
            print("\n" + result)
        else:
            print("\n[ERROR] Future returned None.")
