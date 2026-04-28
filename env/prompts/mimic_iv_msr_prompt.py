CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a patient profile, your task is to find appropriate specialty for the patient. Here are all the specialties in the hospital as below (one specialty per line, in the format of "- specialty"):
- ENT (Ear, Nose, Throat)
- Neurology
- Neurosurgery
- Endocrinology
- Dermatology
- Orthopedics
- Cardiology
- Rheumatology
- Hematology
- Emergency Medicine
- Obstetrics & Gynecology (OB/GYN)
- Urology
- Ophthalmology
- Psychiatry & Mental Health
- Oncology
- Internal Medicine
- Infectious Disease
- General Surgery
- Nephrology

Here are some relevant cases:
{case_prompt}

Now, given the following patient profile:
{task}
Please directly provide the best specialty for the patient in the following format:
\\boxed{{specialty}}
"""


ZERO_SHOT_PROMPT = """Given a patient profile, your task is to find appropriate specialty for the patient. Here are all the specialties in the hospital as below (one specialty per line, in the format of "- specialty"):
- ENT (Ear, Nose, Throat)
- Neurology
- Neurosurgery
- Endocrinology
- Dermatology
- Orthopedics
- Cardiology
- Rheumatology
- Hematology
- Emergency Medicine
- Obstetrics & Gynecology (OB/GYN)
- Urology
- Ophthalmology
- Psychiatry & Mental Health
- Oncology
- Internal Medicine
- Infectious Disease
- General Surgery
- Nephrology
Now, given the following patient profile:
{task}
Please directly provide the best specialty for the patient in the following format:
\\boxed{{specialty}}
"""
