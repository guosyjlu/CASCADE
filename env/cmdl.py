"""
This is the environment file for CMDL.
- Legal decision making: multi-defendant penalty prediction.
[Ref] CMDL: A Large-Scale Chinese Multi-Defendant Legal Judgment Prediction Dataset
"""
import ast
import json
from .base import Env
from .prompts.cmdl_prompt import ZERO_SHOT_PROMPT, CASE_PROMPT, CBR_PROMPT, CASE_PROMPT_REFLEXION

class CMDLEnv(Env):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CASE_PROMPT_REFLEXION = CASE_PROMPT_REFLEXION
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        with open("data/cmdl/cmdl.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
                
    def observe(self):
        return self.dataset[self.index]["task"]
    
    def evaluate(self, generated_text):
        generated_answer = self.extraction(generated_text)
        reward = self.reward_function(generated_answer)
        return generated_answer, reward
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        names = ", ".join(list(self.dataset[self.index]["label"].keys()))
        return self.ZERO_SHOT_PROMPT.format(task=problem, names=names)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        names = ", ".join(list(self.dataset[self.index]["label"].keys()))
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem, names=names)
    
    def get_case_based_prompt_reflexion(self, problem, cases):
        assert self.observe() == problem
        names = ", ".join(list(self.dataset[self.index]["label"].keys()))
        case_prompt = ""
        for case in cases:
            if case["reward"] == 1:
                feedback = "This answer is correct!"
            else:
                feedback = "This answer is wrong!"
            case_prompt += self.CASE_PROMPT_REFLEXION.format(task=case["task"], answer=case["answer"], feedback=feedback)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem, names=names)
    
    def __len__(self):
        return len(self.dataset)
    
    def extraction(self, generated_text):
        try:
            ans_dict = ast.literal_eval(generated_text.strip())
        except Exception:
            ans_dict = None
        return ans_dict

    def reward_function(self, generated_answer):
        if generated_answer is None:
            reward = 0
        else:
            try:
                ground_truth = self.dataset[self.index]["label"]
                assert len(generated_answer) == len(ground_truth)
                for key in ground_truth:
                    assert len(ground_truth[key]) == len(generated_answer[key])
                    for name in ground_truth[key]:
                        assert name in generated_answer[key]
                reward = 1
            except Exception:
                reward = 0
        return reward

    def get_ground_truth(self):
        return self.dataset[self.index]["label"]