CASE_PROMPT = """##Task
{task}
##Interaction trajectory
{trajectory}
"""

CASE_PROMPT_REFLEXION = """##Task
{task}
##Interaction trajectory
{trajectory}
##Feedback
{feedback}
"""

CBR_PROMPT = """You are an expert agent operating in the ALFRED embodied Environment. Please try your best to complete the task exactly following the description.

Here are some relevant cases:
{case_prompt}

Here is the task description:
{task}
Now it’s your turn to take actions. Please directly output the selected action from the admissible action list without any other texts. Let's start!
"""


ZERO_SHOT_PROMPT = """You are an expert agent operating in the ALFRED embodied Environment. Please try your best to complete the task exactly following the description.
Here is the task description:
{task}
Now it’s your turn to take actions. Please directly output the selected action from the admissible action list without any other texts. Let's start!
"""

SUFFIX_PROMPT = """[Admissible Action List] {action_list}
[Action]"""

HISTORY_PROMPT = """[Action] {action}
[Observation] {state}
"""


