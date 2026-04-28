REACT_EXAMPLE = """##Task
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put some spraybottle on toilet.
##Interaction trajectory
[Action] think: To solve the task, I need to find and take a sparybottle, then put it on toilet.
[Observation] OK.
[Action] think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
[Observation] OK.
[Action] go to cabinet 1
[Observation] On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
[Action] go to cabinet 2
[Observation] The cabinet 2 is closed.
[Action] open cabinet 2
[Observation] You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
[Action] think: Now I find a spraybottle (2). Next, I need to take it.
[Observation] OK.
[Action] take spraybottle 2 from cabinet 2
[Observation] You pick up the spraybottle 2 from the cabinet 2.
[Action] think: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.
[Observation] OK.
[Action] go to toilet 1
[Observation] On the toilet 1, you see a soapbottle 2.
[Action] put spraybottle 2 in/on toilet 1
[Observation] You put the spraybottle 2 in/on the toilet 1.
"""


CASE_PROMPT = """##Task
{task}
##Interaction trajectory
{trajectory}
"""


PROMPT = """You are an expert agent operating in the ALFRED embodied Environment. Please try your best to complete the task exactly following the description.

Here are some relevant cases:
{case_prompt}

Here is the task description:
{task}
Now it’s your turn to take actions to interact with the environment. Let's start!
"""

SUFFIX_PROMPT = """Here are the admissible actions in the current step:
[Admissible Action List] {action_list}
Beyond these actions, you may use "think:" action to perform basic reasoning, but only when it is necessary.
ONLY output the action using the specified format, and MUST NOT output any other redundant contents. Format: [Action] action"""

HISTORY_PROMPT = """[Action] {action}
[Observation] {state}
"""


