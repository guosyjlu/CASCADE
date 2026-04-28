CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a patient profile, your task is to determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity):
1: Assign if the patient requires immediate lifesaving intervention.
2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress).
3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care.
4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG).
5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).

Here are some relevant cases:
{case_prompt}

Now, given the following patient profile:
{task}
Please directly provide the triage level for the patient in the following format:
\\boxed{{triage}}
"""


ZERO_SHOT_PROMPT = """Given a patient profile, your task is to determine the triage level based on the Emergency Severity Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 (lowest acuity):
1: Assign if the patient requires immediate lifesaving intervention.
2: Assign if the patient is in a high-risk situation (e.g., confused, lethargic, disoriented, or experiencing severe pain/distress).
3: Assign if the patient requires two or more diagnostic or therapeutic interventions and their vital signs are within acceptable limits for non-urgent care.
4: Assign if the patient requires one diagnostic or therapeutic intervention (e.g., lab test, imaging, or EKG).
5: Assign if the patient does not require any diagnostic or therapeutic interventions beyond a physical exam (e.g., no labs, imaging, or wound care).

Now, given the following patient profile:
{task}
Please directly provide the triage level for the patient in the following format:
\\boxed{{triage}}
"""
