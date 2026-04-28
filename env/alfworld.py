"""
This is the environment file for ALFWorld.
- Seuquential decision making: embodied task.
[Ref] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning
"""
import json
from .base import SeqEnv
from .prompts.alfworld_prompt import *
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from textworld.gym.envs import TextworldGymEnv

class ALFWorldEnv(SeqEnv):
    def __init__(self, alfworld_path="<YOUR_ALFWORLD_PATH>"):
        super().__init__()
        self.alfworld_path = alfworld_path
        self.dataset = []
        self.alfworld_env = self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CASE_PROMPT_REFLEXION = CASE_PROMPT_REFLEXION
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        with open("data/alfworld/alfworld.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)

        game_path = [sample["game_file"] for sample in self.dataset]
        game_files = [f"{self.alfworld_path}/{path}" for path in game_path]
        
        domain_randomization = False
        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env = TextworldGymEnv(game_files, request_infos, max_episode_steps=30, wrappers=wrappers)
        env._gamefiles_iterator = iter(env.gamefiles)
        return env
    
    def reset(self):
        self.alfworld_env._gamefiles_iterator = iter(self.alfworld_env.gamefiles)
        self.index = 0
        return False
                
    def observe(self):
        return self.dataset[self.index]["task"]
    
    def reset_env(self):
        obs, info = self.alfworld_env.reset()
        assert obs.split("-= Welcome to TextWorld, ALFRED! =-\n\n")[1] == self.observe()
        self.action_list = info["admissible_commands"]
        return None, 0, False, self.action_list
    
    def interact(self, generated_text):
        action = generated_text.split("\n")[0].strip()
        if action not in self.action_list:
            obs, reward, stop, infos = self.alfworld_env.step("look")
            obs = "The action is illegal. Please retry."
        else:
            obs, reward, stop, infos = self.alfworld_env.step(action)
            self.trajectory_prompt += HISTORY_PROMPT.format(state=obs, action=action)
        self.action_list = infos["admissible_commands"]
        self.history_prompt += HISTORY_PROMPT.format(state=obs, action=action)
        return obs, reward, stop, self.action_list
    
    def get_trajectory(self):
        return self.trajectory_prompt
    
    def get_prompt(self, obs, action_list):
        if obs is None:
            self.history_prompt = ""
            self.trajectory_prompt = ""
        suffix = SUFFIX_PROMPT.format(action_list=action_list)
        prompt = self.history_prompt + suffix
        return prompt
            
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        return self.ZERO_SHOT_PROMPT.format(task=problem)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], trajectory=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)

    def get_case_based_prompt_reflexion(self, problem, cases):
        assert self.observe() == problem
        case_prompt = ""
        for case in cases:
            if case["reward"] == 1:
                feedback = "This trajectory fails!"
            else:
                feedback = "This trajectory is successful!"
            case_prompt += self.CASE_PROMPT_REFLEXION.format(task=case["task"], trajectory=case["answer"], feedback=feedback)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)
    
    def __len__(self):
        return len(self.dataset)
    
