"""
This is the environment file for ScienceWorld, specifically adapted to ReAct.
- Seuquential decision making: embodied task.
[Ref] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning
"""
import json
from .base import SeqEnv
from .prompts.scienceworld_react_prompt import *
from scienceworld import ScienceWorldEnv

task_step_map = {
    "use-thermometer": 30,
    "test-conductivity": 30,
    "test-conductivity-of-unknown-substances": 30,
    "find-animal": 15,
    "find-living-thing": 15,
    "find-non-living-thing": 15,
    "find-plant": 15,
    "grow-plant": 30,
    "lifespan-longest-lived": 10,
    "lifespan-longest-lived-then-shortest-lived": 12,
    "lifespan-shortest-lived": 10
}

def equivalent_multiline(a: str, b: str) -> bool:
    """
        Analyze whether two strings are fundamentally equivalent (if all the order-agnostic contents are the same).
    """
    def normalize(s: str):
        return sorted(
            line.strip() for line in s.strip().splitlines() if line.strip()
        )
    return normalize(a) == normalize(b)

class SeqScienceWorldReActEnv(SeqEnv):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.scienceworld_env = self.init_env()
        self.CASE_PROMPT = CASE_PROMPT
        self.PROMPT_TEMPLATE = PROMPT
        self.REACT_EXAMPLE = REACT_EXAMPLE
        
    def init_env(self):
        with open("data/scienceworld/scienceworld.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
        env = ScienceWorldEnv(None, None)
        return env
    
    def reset(self):
        self.index = 0
        return False
                
    def observe(self):
        return self.dataset[self.index]["task"]
    
    def reset_env(self):
        taskName = self.dataset[self.index]["taskName"]
        variationIdx = self.dataset[self.index]["variationIdx"]
        self.scienceworld_env.load(taskName, variationIdx, "easy", False)
        obs, info = self.scienceworld_env.reset()
        # Check consistency
        desc = self.scienceworld_env.get_task_description()
        look = self.scienceworld_env.look()
        inventory = self.scienceworld_env.inventory()
        assert equivalent_multiline(self.observe(), f"{desc}\n{look}{inventory}")
        # Record steps
        self.iter = 0
        self.max_iter = task_step_map[taskName] + 5     # More budgets for ReAct think action
        return None, 0, False, None
    
    def interact(self, generated_text: str):
        generated_text = generated_text.split("\n")[0].strip()
        if not generated_text.startswith("[Action]"):
            action = generated_text.split("\n")[0].strip()
            obs, _, stop, infos = self.scienceworld_env.step("look around")
            obs = "The action must start with [Action]! Please retry."
        else:
            action = generated_text.split("[Action]")[1].strip()
            if action.startswith("think:"):
                obs, _, stop, infos = self.scienceworld_env.step("look around")
                obs = "OK."
            else:
                obs, _, stop, infos = self.scienceworld_env.step(action)
            if "No known action matches that input." not in obs:
                self.trajectory_prompt += HISTORY_PROMPT.format(state=obs, action=action)
        # Process reward: reward = 1, only if score = 100.
        self.history_prompt += HISTORY_PROMPT.format(state=obs, action=action)
        reward = int(infos["score"] == 100)
        # Process stop
        self.iter += 1
        if self.iter == self.max_iter:
            stop = True
        return obs, reward, stop, None
    
    def get_trajectory(self):
        return self.trajectory_prompt
    
    def get_prompt(self, obs, action_list):
        if obs is None:
            self.history_prompt = ""
            self.trajectory_prompt = ""
        suffix = SUFFIX_PROMPT.format()
        prompt = self.history_prompt + suffix
        return prompt
            
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        case_prompt = self.REACT_EXAMPLE
        return self.PROMPT_TEMPLATE.format(case_prompt=case_prompt, task=problem)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], trajectory=case["answer"])
        return self.PROMPT_TEMPLATE.format(case_prompt=case_prompt, task=problem)
    
    def __len__(self):
        return len(self.dataset)
    
