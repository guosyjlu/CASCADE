"""
Base environemnt class.
"""
from abc import ABC, abstractmethod

class Env(ABC):
    def __init__(self):
        self.type = "single-turn"
        self.index = 0
    
    def step(self):
        self.index += 1
        if self.index >= len(self):
            return True
        else:
            return False
    
    def reset(self):
        self.index = 0
        return False
    
    @abstractmethod
    def __len__(self):
        """Return the length of the environment (must be implemented by subclass)."""
        pass
    
    @abstractmethod
    def observe(self):
        """Return the query of each step (must be implemented by subclass)."""
        pass
    
    @abstractmethod
    def evaluate(self, generated_text):
        """Evaluate the generated answer, return the reward and generated answer (must be implemented by subclass)."""
        pass
    
    @abstractmethod
    def get_zero_shot_prompt(self, problem):
        """Return the zero-shot prompt of the given problem (must be implemented by subclass)."""
        pass
    
    @abstractmethod
    def get_case_based_prompt(self, problem, cases):
        """Return the case-based prompt of the given problem and the given case (must be implemented by subclass)."""
        pass


class SeqEnv(ABC):
    def __init__(self):
        self.type = "multi-turn"
        self.index = 0
        
    def step(self):
        self.index += 1
        if self.index >= len(self):
            return True
        else:
            return False

class PlannerExecutorEnv(ABC):
    def __init__(self):
        self.type = "planner-executor"
        self.index = 0
        
    def step(self):
        self.index += 1
        if self.index >= len(self):
            return True
        else:
            return False