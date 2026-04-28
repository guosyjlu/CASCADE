CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a patient profile, your task is to diagnose the patient. Here are all the possible diagnoses as below (one diagnosis per line, in the format of "- diagnosis"):
- Acute COPD exacerbation / infection
- Acute dystonic reactions
- Acute laryngitis
- Acute otitis media
- Acute pulmonary edema
- Acute rhinosinusitis
- Allergic sinusitis
- Anaphylaxis
- Anemia
- Atrial fibrillation
- Boerhaave
- Bronchiectasis
- Bronchiolitis
- Bronchitis
- Bronchospasm / acute asthma exacerbation
- Chagas
- Chronic rhinosinusitis
- Cluster headache
- Croup
- Ebola
- Epiglottitis
- GERD
- Guillain-Barré syndrome
- HIV (initial infection)
- Influenza
- Inguinal hernia
- Larygospasm
- Localized edema
- Myasthenia gravis
- Myocarditis
- PSVT
- Pancreatic neoplasm
- Panic attack
- Pericarditis
- Pneumonia
- Possible NSTEMI / STEMI
- Pulmonary embolism
- Pulmonary neoplasm
- SLE
- Sarcoidosis
- Scombroid food poisoning
- Spontaneous pneumothorax
- Spontaneous rib fracture
- Stable angina
- Tuberculosis
- URTI
- Unstable angina
- Viral pharyngitis
- Whooping cough

Here are some relevant cases:
{case_prompt}

Now, given the following patient profile:
{task}
Please directly provide the diagnosis for the patient in the following format:
\\boxed{{diagnosis}}
"""


ZERO_SHOT_PROMPT = """Given a patient profile, your task is to diagnose the patient. Here are all the possible diagnoses as below (one diagnosis per line, in the format of "- diagnosis"):
- Acute COPD exacerbation / infection
- Acute dystonic reactions
- Acute laryngitis
- Acute otitis media
- Acute pulmonary edema
- Acute rhinosinusitis
- Allergic sinusitis
- Anaphylaxis
- Anemia
- Atrial fibrillation
- Boerhaave
- Bronchiectasis
- Bronchiolitis
- Bronchitis
- Bronchospasm / acute asthma exacerbation
- Chagas
- Chronic rhinosinusitis
- Cluster headache
- Croup
- Ebola
- Epiglottitis
- GERD
- Guillain-Barré syndrome
- HIV (initial infection)
- Influenza
- Inguinal hernia
- Larygospasm
- Localized edema
- Myasthenia gravis
- Myocarditis
- PSVT
- Pancreatic neoplasm
- Panic attack
- Pericarditis
- Pneumonia
- Possible NSTEMI / STEMI
- Pulmonary embolism
- Pulmonary neoplasm
- SLE
- Sarcoidosis
- Scombroid food poisoning
- Spontaneous pneumothorax
- Spontaneous rib fracture
- Stable angina
- Tuberculosis
- URTI
- Unstable angina
- Viral pharyngitis
- Whooping cough
Now, given the following patient profile:
{task}
Please directly provide the diagnosis for the patient in the following format:
\\boxed{{diagnosis}}
"""
