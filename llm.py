"""
This file provides two LLM interfaces:
(1) OpenAI compatible interface: query_openai
(2) vLLM deployed models with OpenAI compatible interface: query_vllm_openai
"""
import os
import time
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "none"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "none"

def query_openai(prompt, model="gemini-2.0-flash", max_retries=10, retry_delay=30):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.05,
                top_p=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}.")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                return None

def query_vllm_openai(prompt, model, server, port=8000, max_retries=10, retry_delay=30, enable_thinking=False, max_completion_tokens=None):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=f"http://{server}:{port}/v1")
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_completion_tokens=max_completion_tokens,
                temperature=0.1,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError("Failed to get response from vLLM after multiple attempts.")

def query_openai_message(messages, model="gpt-4o-mini", tools=None, max_retries=10, retry_delay=30):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    retries = 0
    while retries < max_retries:
        try:
            if tools is not None:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=False,
                    temperature=0.05,
                    top_p=0.8,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=0.05,
                    top_p=0.8,
                )
            msg = response.choices[0].message
            raw_calls = getattr(msg, "tool_calls", None)
            tool_calls = None
            if raw_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_calls
                ]
            return {"content": msg.content, "tool_calls": tool_calls}
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}.")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                return None

def query_vllm_openai_message(messages, server, port=8000, model="qwen3-32b", tools=None, tool_choice="auto", enable_thinking=False, max_retries=10, retry_delay=30):
    client = OpenAI(api_key="none", base_url=f"http://{server}:{port}/v1")
    retries = 0
    while retries < max_retries:
        try:
            if tools is not None:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=False,
                    temperature=0.1,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=0.1,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )
            msg = response.choices[0].message
            raw_calls = getattr(msg, "tool_calls", None)
            tool_calls = None
            if raw_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_calls
                ]
            return {"content": msg.content, "tool_calls": tool_calls}
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}.")
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                return None


if __name__ == '__main__':
    print(query_openai("Hello"))