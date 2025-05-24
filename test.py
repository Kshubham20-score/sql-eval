from openai import OpenAI
from typing import Callable
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

client_openai = OpenAI(base_url="https://node4-api.staging.greenjello.io/v1", api_key="EMPTY")

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
                model="Qwen/Qwen2.5-Coder-14B",
                max_completion_tokens=100,
                temperature=0.1,
                stop=[],
                seed=0,
                store=True,
                metadata=None,
            )

print( response.choices[0].message.content)


#from utils.llm import chat_openai
import sqlparse

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
        if self.model in LLM_COSTS_PER_TOKEN:
            model_name = self.model
        else:
            model_name = None
            potential_model_names = []

            for mname in LLM_COSTS_PER_TOKEN.keys():
                if mname in self.model:
                    potential_model_names.append(mname)

            if len(potential_model_names) > 0:
                model_name = max(potential_model_names, key=len)

        if model_name:
            self.cost_in_cents = (
                self.input_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]
                + self.output_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"]
            ) * 100

def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 600,
    temperature: float = 0.1,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
    base_url: str=None,
) -> LLMResponse:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    """
    from openai import OpenAI

    #if base_url:
    #    print("check1")
    #    client_openai = OpenAI(base_url=base_url, api_key="EMPTY")
    #else:
    #    client_openai = OpenAI()
    t = time.time()
    if model.startswith("o"):
        if messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = sys_msg + messages[0]["content"]

        response = client_openai.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            store=store,
            metadata=metadata,
            timeout=timeout,
        )
    else:
        if response_format or json_mode:
            response = client_openai.beta.chat.completions.parse(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                response_format=(
                    {"type": "json_object"} if json_mode else response_format
                ),
                seed=seed,
                store=store,
                metadata=metadata,
            )
        else:
            print("check2")
            response = client_openai.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                seed=seed,
                store=store,
                metadata=metadata,
            )

    if response_format and not model.startswith("o1"):
        content = response.choices[0].message.parsed
    else:
        content = response.choices[0].message.content
    print(content)
    if response.choices[0].finish_reason == "length":
        print("Max tokens reached")
        #raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        print("Empty response")
        raise Exception("No response")
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        output_tokens_details=response.usage.completion_tokens_details,
    )


response = chat_openai(messages=messages, model="Qwen/Qwen2.5-Coder-14B", temperature=0.1, base_url="https://node4-api.staging.greenjello.io/v1")
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
