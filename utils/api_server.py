import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.lora.request import LoRARequest
from vllm import __version__ as vllm_version

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

# This is a fork of https://github.com/vllm-project/vllm/blob/0650e5935b0f6af35fb2acf71769982c47b804d7/vllm/entrypoints/api_server.py
# with the following changes:
# - remove the prompt from response. we only return the generated output to avoid parsing errors when including the prompt.
# - don't add special_tokens (bos/eos) and only add it if it's missing from the prompt
# You can start it similar to how you would with the usual vllm api server:
# ```
# python3 utils/api_server.py \
#   --model "${model_path}" \
#   --tensor-parallel-size 4 \
#   --dtype float16 \
#   --max-model-len 4096 \
#   --port 5000 \
#   --gpu-memory-utilization 0.90 \
#   --enable-lora \
#   --max-lora-rank 64 \


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sql_lora_path = request_dict.pop("sql_lora_path", None)
    request_dict.pop("sql_lora_name", None)
    lora_request = (
        LoRARequest(lora_name="sql_adapter", lora_int_id=1, lora_path=sql_lora_path)
        if sql_lora_path
        else None
    )
    if vllm_version >= "0.6.2":
        # remove use_beam_search  if present as it's no longer supported
        # see https://github.com/vllm-project/vllm/releases/tag/v0.6.2
        if "use_beam_search" in request_dict:
            request_dict.pop("use_beam_search")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    tokenizer = await engine.get_tokenizer()
    prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    if prompt_token_ids[0] != tokenizer.bos_token_id:
        prompt_token_ids = [tokenizer.bos_token_id] + prompt_token_ids

    if vllm_version >= "0.6.3":
        from vllm import TokensPrompt

        results_generator = engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
    elif vllm_version >= "0.4.2":
        results_generator = engine.generate(
            inputs={"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
    else:
        results_generator = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
        )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = []
    for output in final_output.outputs:
        text_outputs.append(output.text)

    try:
        logprobs = [output.logprobs for output in final_output.outputs]
    except Exception as e:
        logprobs = []
        print(e)
        print("Could not extract logprobs")

    logprobs = logprobs[0]
    logprobs_json = []
    if logprobs:
        for item in logprobs:
            # do this to make our response JSON serializable
            item = {key: value.__dict__ for key, value in item.items()}
            logprobs_json.append(item)

    ret = {"text": text_outputs, "logprobs": logprobs_json}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
