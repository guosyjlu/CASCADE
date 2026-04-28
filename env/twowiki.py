"""
This is the environment file for 2Wiki.
- 2Wiki: deep search task.
"""
import re
import json
from .base import PlannerExecutorEnv
from .prompts.twowiki_prompt import *
from llm import query_openai, query_vllm_openai_message, query_openai_message

from pathlib import Path
from typing import Any, Dict, List
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\s\S]*}", text)
    return m.group(0) if m else text

MAX_TURNS_MEMORY = 50

class TwoWikiEnv(PlannerExecutorEnv):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.tools_schema = None
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.init_env()
        
    def init_env(self):
        # Load dataset
        with open("data/2wiki/2wiki.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
        
    async def connect_to_servers(self, mode="local"):
        if mode == "local":
            scripts = ["env/deepsearch/rag_mcp.py"]
        elif mode == "online":
            scripts = ["env/deepsearch/serper_snippet_mcp.py"]
        else:
            raise NotImplementedError("The specified mode is not supported yet.")
        
        for script in scripts:
            path = Path(script)
            cmd = "python"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
        print("Connected tools:", list(self.sessions.keys()))
        
        self.tools_schema = await self._tools_schema()
        print(self.tools_schema)
    
    async def _tools_schema(self) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        self.shared_history = []
        self._add_to_history("user", f"{DEEP_RESEARCH_PROMPT}\nQuestion: {problem}\nNow, it's your turn!")
        msgs = self.shared_history.copy()
        return msgs
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        self.shared_history = []
        case_prompt = ""
        for case in cases:
            case_prompt += DEEP_RESEARCH_CASE_PROMPT.format(task=case["query"], trajectory=case["answer"])
        case_based_prompt = DEEP_RESEARCH_CBR_PROMPT.format(case_prompt=case_prompt)
        self._add_to_history("user", f"{DEEP_RESEARCH_PROMPT}\n{case_based_prompt}\nQuestion: {problem}")
        msgs = self.shared_history.copy()
        return msgs
    
    def invoke_llm(self, msgs, args, tool_choice):
        if args.serving_mode == "vllm":
            response = query_vllm_openai_message(msgs, model=args.llm, server=args.server, port=args.port, tools=self.tools_schema, tool_choice=tool_choice)
        else:
            response = query_openai_message(msgs, model=args.llm, tools=self.tools_schema)
        return response
    
    async def interact(self, msgs, args):
        MAX_TOOL_CALLS = 5
        TOOL_COUNT = 0
        final_answer = ""
        trajectory = ""
        summary = ""
        while TOOL_COUNT <= MAX_TOOL_CALLS:
            content = self.invoke_llm(msgs, args=args, tool_choice="auto")
            if content["content"]:
                final_answer = str(content["content"])
                trajectory += f"[Assistant] {final_answer}"
                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "summary": summary,
                }
            else:
                for call in content.get("tool_calls") or []:
                    t_name = call["function"]["name"]
                    t_args = json.loads(call["function"].get("arguments") or "{}")
                    if t_name in self.sessions:
                        session = self.sessions[t_name]
                        result_msg = await session.call_tool(t_name, t_args)
                        result_text = str(result_msg.content)                        
                    else:
                        result_text = "The specified tool is not in the given tool set."
                    
                    msgs.extend(
                        [
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                        ]
                    )
                    TOOL_COUNT += 1
                    if t_name in self.sessions:
                        # Add tool summary
                        if t_name == "crawl_extract":
                            filtered_args = dict(t_args)
                            filtered_args.pop("url", None)
                            summary += f"[Tool] {t_name}, {filtered_args}\n"
                        else:
                            summary += f"[Tool] {t_name}, {t_args}\n"
                        # Add trajectory
                        trajectory += "[Assistant] Tool calls: " + json.dumps({
                            "function": t_name,
                            "arguments": t_args
                        }) + "\n"
                        trajectory += "[Tool Results] " + result_text
                    
        final_answer = "Exceeded maximum tool calls."
        print("Exceed maximum tool calls.")
        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "summary": summary,
        }
    
    def _add_to_history(self, role: str, content: str):
        self.shared_history.append({"role": role, "content": content})
        if len(self.shared_history) > MAX_TURNS_MEMORY:
            self.shared_history.pop(0)
    
    def reward_function(self, generated_answer):
        ground_truth = self.dataset[self.index]["label"]
        judgement = self._LLM_as_judge(
            query=self.dataset[self.index]["task"],
            ground_truth=ground_truth,
            pred_answer=generated_answer
         )
        if judgement["judgement"] == "correct":
            reward = 1
        else:
            reward = 0
        return reward, judgement
    
    def reset(self):
        self.index = 0
        return False
                
    def observe(self):
        return self.dataset[self.index]["task"]

    def __len__(self):
        return len(self.dataset)
    
    def _LLM_as_judge(self, query, ground_truth, pred_answer, model_name="gpt-4o-mini", retry=10):
        count = 0
        while count <= retry:
            try:
                prompt = PROMPT_TPL.format(question=query, gt_answer=ground_truth, pred_answer=pred_answer)
                content = query_openai(prompt, model=model_name)
                content = _strip_fences(content)
                data = json.loads(content)
                judgement = data["judgement"].lower().strip()
                assert judgement in ("correct", "incorrect")
                rationale = str(data.get("rationale", ""))
                return {"judgement": judgement, "rationale": rationale}
            except Exception as e:
                print(f"LLM As Judge - Failed to parse LLM response: {e}")
                print(content)
                count += 1
                # Try to fix the json output
                try:
                    prompt = JSON_FIX_PROMPT.format(json=content)
                    content = query_openai(prompt, model=model_name)
                    content = _strip_fences(content)
                    data = json.loads(content)
                    judgement = data["judgement"].lower().strip()
                    assert judgement in ("correct", "incorrect")
                    rationale = str(data.get("rationale", ""))
                    return {"judgement": judgement, "rationale": rationale}
                except Exception as e:
                    pass     
        return {"judgement": "incorrect", "rationale": "The LLM judge failed to give a valid response after multiple retries."}
    
    async def cleanup(self):
        await self.exit_stack.aclose()
    