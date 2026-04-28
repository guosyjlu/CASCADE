CASE_PROMPT = """[Task] {task}
[Answer] {answer}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] {answer}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given the description of a legal case, your task is to predict the penalty judgement for each defendant.
There are six different types of penalties: Surveillance, Detention, Imprisonment, Death Penalty, Life Imprisonment, Fine. Also, some defendants may have multiple penalties.

Here are some relevant cases:
{case_prompt}

Now, given the description of a legal case:
{task}
Now, please directly provide the penality list for each defendant ({names}) with the following format. Do not output any explanations.
{{'name1': penalty_list1, 'name2': penaly_list2, ...}}
"""


ZERO_SHOT_PROMPT = """Given the description of a legal case, your task is to predict the penalty judgement for each defendant.
There are six different types of penalties: Surveillance, Detention, Imprisonment, Death Penalty, Life Imprisonment, Fine. Also, some defendants may have multiple penalties.
Now, given the description of a legal case:
{task}
Now, please directly provide the penality list for each defendant ({names}) with the following format. Do not output any explanations.
{{'name1': penalty_list1, 'name2': penaly_list2, ...}}
"""
