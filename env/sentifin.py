"""
This is the environment file for SEntFiN.
- Financial: entity-level financial sentiment analysis.
[Ref] SILC-EFSA: Self-aware In-context Learning Correction for Entity-level Financial Sentiment Analysis
"""
import ast
import json
from .base import Env
from .prompts.sentifin_prompt import ZERO_SHOT_PROMPT, CASE_PROMPT, CBR_PROMPT, CASE_PROMPT_REFLEXION

class SEntFiNEnv(Env):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CASE_PROMPT_REFLEXION = CASE_PROMPT_REFLEXION
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        with open("data/sentifin/sentifin.jsonl", encoding="utf-8") as file:
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
        return self.ZERO_SHOT_PROMPT.format(task=problem)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            answer = "\n".join(str(case["answer"]))
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=answer)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)
    
    def get_case_based_prompt_reflexion(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            answer = "\n".join(str(case["answer"]))
            if case["reward"] == 1:
                feedback = "This answer is correct!"
            else:
                feedback = "This answer is wrong!"
            case_prompt += self.CASE_PROMPT_REFLEXION.format(task=case["task"], answer=answer, feedback=feedback)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)
    
    def __len__(self):
        return len(self.dataset)
    
    def extraction(self, generated_text: str):
        try:
            answer = generated_text.splitlines()
            ans_dict = [ast.literal_eval(line) for line in answer if line.strip()]
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
                for i in range(len(ground_truth)):
                    assert ground_truth[i]["entity"] == generated_answer[i]["entity"]
                    assert ground_truth[i]["label"] == generated_answer[i]["label"]
                reward = 1
            except Exception:
                reward = 0
        return reward

    def get_ground_truth(self):
        return self.dataset[self.index]["label"]