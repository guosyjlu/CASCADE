CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a piece of log, your task is to indicate the type of the alert. All the 6 possible types: EventTrap, communicationsAlarm, qualityOfServiceAlarm, processingErrorAlarm, equipmentAlarm, environmentalAlarm.

Here are some relevant cases:
{case_prompt}

Now, given the log:
{task}
Please directly provide the alert type in the following format:
\\boxed{{type}}
"""


ZERO_SHOT_PROMPT = """Given a piece of log, your task is to indicate the type of the alert. All the 6 possible types: EventTrap, communicationsAlarm, qualityOfServiceAlarm, processingErrorAlarm, equipmentAlarm, environmentalAlarm.
Now, given the log:
{task}
Please directly provide the alert type in the following format:
\\boxed{{type}}
"""
