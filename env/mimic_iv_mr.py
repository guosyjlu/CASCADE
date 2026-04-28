"""
This is the environment file for MIMIC-IV-MR.
- Medical decision making: medicine recommendation.
[Ref.1] Large Language Model Distilling Medication Recommendation Model
[Ref.2] https://mimic.mit.edu/
"""
import ast
import json
from .base import Env
from .prompts.mimic_iv_mr_prompt import ZERO_SHOT_PROMPT, CASE_PROMPT, CBR_PROMPT, CASE_PROMPT_REFLEXION

class MIMIC_IV_MREnv(Env):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CASE_PROMPT_REFLEXION = CASE_PROMPT_REFLEXION
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        with open("data/mimic-iv-mr/mimic-iv-mr.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
                
    def observe(self):
        return self.dataset[self.index]["task"]
    
    def evaluate(self, generated_text):
        generated_answer = self.extraction(generated_text)
        reward = self.reward_function(generated_answer)
        if reward == 1:
            generated_answer = self.dataset[self.index]["label"]
        return generated_answer, reward
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        return self.ZERO_SHOT_PROMPT.format(task=problem)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)

    def get_case_based_prompt_reflexion(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            if case["reward"] == 1:
                feedback = "This answer is correct!"
            else:
                feedback = "This answer is wrong!"
            case_prompt += self.CASE_PROMPT_REFLEXION.format(task=case["task"], answer=case["answer"], feedback=feedback)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)
    
    def __len__(self):
        return len(self.dataset)
    
    def extraction(self, generated_text):
        try:
            answer = ast.literal_eval(generated_text.strip())
        except Exception:
            answer = None
        return answer

    def reward_function(self, generated_answer):
        if generated_answer is None:
            reward = 0
        else:
            ground_truth = self.dataset[self.index]["label"]
            generated_answer = set(generated_answer)
            ground_truth = set(ground_truth)
            intersection = generated_answer & ground_truth
            union = generated_answer | ground_truth
            jacob_sim = len(intersection) / len(union)
            reward = int(jacob_sim >= 0.2)
        return reward
    
    def get_ground_truth(self):
        return self.dataset[self.index]["label"]
