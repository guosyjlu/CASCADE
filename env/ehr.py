"""
This is the environment file for EHR.
"""
import json
from .base import SeqEnv
from .prompts.ehr_prompt import *
from .ehragent.execute import run_code_with_limited_time

def convert_number_2f(text: str):
    try:
        num = float(text)
        return f"{num:.2f}"
    except (ValueError, TypeError):
        return text


class EHREnv(SeqEnv):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.init_env()
        self.internal_step_count = 0
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        with open("data/ehr/ehr.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
    
    def reset(self):
        self.index = 0
        return False
                
    def observe(self):
        return self.dataset[self.index]["task"]
    
    def reset_env(self):
        self.internal_step_count = 0
        return None, 0, False, None
    
    def reward_function(self, prediction):
        ground_truth = self.dataset[self.index]["label"]
        
        # Covert yes / no type question
        if prediction == "True":
            prediction = "1"
        if prediction == "False":
            prediction = "0"
        
        prediction = convert_number_2f(prediction)
        ground_truth = convert_number_2f(ground_truth)
        reward = (prediction == ground_truth)
        return reward
    
    def interact(self, generated_text):
        self.internal_step_count += 1
        script = "# The original code is omitted due to wrong output format."
        try:
            script = generated_text.split('```python')[-1].split('```')[0].strip()
            success, answer = run_code_with_limited_time(script)
        except Exception as e:
            success = False
            answer = "Please use ```python``` to format the generated Python code script."
        
        obs = answer
        if success:
            stop = True
            reward = self.reward_function(answer)
        elif self.internal_step_count >= 5:
            stop = True
            reward = 0
        else:
            stop = False
            reward = 0
        
        self.trajectory_prompt = script
        
        self.history_prompt = HISTORY_PROMPT.format(state=obs, action=script)
        return obs, reward, stop, None
    
    def get_trajectory(self):
        return self.trajectory_prompt
    
    def get_prompt(self, obs, action_list):
        if obs is None:
            self.history_prompt = ""
            suffix = ""
        else:
            suffix = SUFFIX_PROMPT
        prompt = self.history_prompt + suffix
        return prompt
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        return self.ZERO_SHOT_PROMPT.format(task=problem)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)
    
    def __len__(self):
        return len(self.dataset)
    
