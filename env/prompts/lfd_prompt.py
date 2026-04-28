CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a set of log informations, your task is to determine what type of fault it belongs to. Here are three types of fault: CPU, Memory, Hardware.

Here are some relevant cases:
{case_prompt}

Now, given the log informations:
{task}
Please directly provide the fault type in the following format:
\\boxed{{type}}
"""


ZERO_SHOT_PROMPT = """Given a set of log informations, your task is to determine what type of fault it belongs to. Here are three types of fault: CPU, Memory, Hardware.
Now, given the log informations:
{task}
Please directly provide the fault type in the following format:
\\boxed{{type}}
"""
