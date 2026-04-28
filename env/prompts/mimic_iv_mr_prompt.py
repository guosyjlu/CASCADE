CASE_PROMPT = """[Task] {task}
[Answer] {answer}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] {answer}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a patient profile, your task is to prescribe appropriate drugs for the patient. Here are all the available drugs in the hospital (splitted by "\n"):
other diuretics
other sex hormones and modulators of the genital system
agents for treatment of hemorrhoids and anal fissures for topical use
iron preparations
blood glucose lowering drugs, excl. insulins
drugs for treatment of tuberculosis
other antineoplastic agents
sulfonamides and trimethoprim
antimycotics for systemic use
anesthetics, local
belladonna and derivatives, plain
other cardiac preparations
anti-dementia drugs
other plain vitamin preparations
other systemic drugs for obstructive airway diseases
antiadrenergic agents, peripherally acting
antiglaucoma preparations and miotics
drugs for constipation
parasympathomimetics
plant alkaloids and other natural products
antigout preparations
antimetabolites
decongestants and other nasal preparations for topical use
low-ceiling diuretics, excl. thiazides
direct acting antivirals
digestives, incl. enzymes
all other therapeutic products
expectorants, excl. combinations with cough suppressants
agents against amoebiasis and other protozoal diseases
androgens
other dermatological preparations
beta blocking agents
posterior pituitary lobe hormones
antimalarials
cytotoxic antibiotics and related substances
thyroid preparations
immunosuppressants
antiemetics and antinauseants
estrogens
anticholinergic agents
hypnotics and sedatives
low-ceiling diuretics, thiazides
antithrombotic agents
vitamin b12 and folic acid
intestinal antiinfectives
corticosteroids for systemic use, plain
drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)
antiarrhythmics, class i and iii
stomatological preparations
insulins and analogues
aminoglycoside antibacterials
anxiolytics
anesthetics, general
tetracyclines
potassium-sparing agents
other gynecologicals
antacids
vitamin k and other hemostatics
drugs used in benign prostatic hypertrophy
psychostimulants, agents used for adhd and nootropics
other antibacterials
selective calcium channel blockers with mainly vascular effects
progestogens
drugs used in addictive disorders
beta-lactam antibacterials, penicillins
antipropulsives
other agents acting on the renin-angiotensin system
other beta-lactam antibacterials
antithyroid preparations
vitamin a and d, incl. combinations of the two
selective calcium channel blockers with direct cardiac effects
ace inhibitors, plain
drugs for functional gastrointestinal disorders
lipid modifying agents, plain
anti-parathyroid agents
ectoparasiticides, incl. scabicides
antipruritics, incl. antihistamines, anesthetics, etc.
antifungals for topical use
antiepileptics
cardiac glycosides
vasodilators used in cardiac diseases
nasal decongestants for systemic use
chemotherapeutics for topical use
antibiotics for topical use
adrenergics, inhalants
antiinflammatory and antirheumatic products, non-steroids
bile therapy
alkylating agents
other mineral supplements
other nervous system drugs
dopaminergic agents
anti-acne preparations for topical use
cardiac stimulants excl. cardiac glycosides
irrigating solutions
antihistamines for systemic use
hormone antagonists and related agents
urologicals
other analgesics and antipyretics
antipsychotics
muscle relaxants, centrally acting agents
vitamin b1, plain and in combination with vitamin b6 and b12
antimigraine preparations
angiotensin ii receptor blockers (arbs), plain
hormonal contraceptives for systemic use
antidepressants
hypothalamic hormones
corticosteroids, plain
calcium
potassium
intestinal antiinflammatory agents
propulsives
macrolides, lincosamides and streptogramins
arteriolar smooth muscle, agents acting on
muscle relaxants, peripherally acting agents
quinolone antibacterials
antiadrenergic agents, centrally acting
antifibrinolytics
high-ceiling diuretics
bacterial vaccines
opioids
glycogenolytic hormones
cough suppressants, excl. combinations with expectorants

Here are some relevant cases:
{case_prompt}

Now, given the patient profile:
{task}
Please directly prescribe a appropriate set of drugs for the patient in the following format:
["drug1", "drug2", "drug3", ...]
"""


ZERO_SHOT_PROMPT = """Given a patient profile, your task is to prescribe appropriate drugs for the patient. Here are all the available drugs in the hospital (splitted by "\n"):
other diuretics
other sex hormones and modulators of the genital system
agents for treatment of hemorrhoids and anal fissures for topical use
iron preparations
blood glucose lowering drugs, excl. insulins
drugs for treatment of tuberculosis
other antineoplastic agents
sulfonamides and trimethoprim
antimycotics for systemic use
anesthetics, local
belladonna and derivatives, plain
other cardiac preparations
anti-dementia drugs
other plain vitamin preparations
other systemic drugs for obstructive airway diseases
antiadrenergic agents, peripherally acting
antiglaucoma preparations and miotics
drugs for constipation
parasympathomimetics
plant alkaloids and other natural products
antigout preparations
antimetabolites
decongestants and other nasal preparations for topical use
low-ceiling diuretics, excl. thiazides
direct acting antivirals
digestives, incl. enzymes
all other therapeutic products
expectorants, excl. combinations with cough suppressants
agents against amoebiasis and other protozoal diseases
androgens
other dermatological preparations
beta blocking agents
posterior pituitary lobe hormones
antimalarials
cytotoxic antibiotics and related substances
thyroid preparations
immunosuppressants
antiemetics and antinauseants
estrogens
anticholinergic agents
hypnotics and sedatives
low-ceiling diuretics, thiazides
antithrombotic agents
vitamin b12 and folic acid
intestinal antiinfectives
corticosteroids for systemic use, plain
drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)
antiarrhythmics, class i and iii
stomatological preparations
insulins and analogues
aminoglycoside antibacterials
anxiolytics
anesthetics, general
tetracyclines
potassium-sparing agents
other gynecologicals
antacids
vitamin k and other hemostatics
drugs used in benign prostatic hypertrophy
psychostimulants, agents used for adhd and nootropics
other antibacterials
selective calcium channel blockers with mainly vascular effects
progestogens
drugs used in addictive disorders
beta-lactam antibacterials, penicillins
antipropulsives
other agents acting on the renin-angiotensin system
other beta-lactam antibacterials
antithyroid preparations
vitamin a and d, incl. combinations of the two
selective calcium channel blockers with direct cardiac effects
ace inhibitors, plain
drugs for functional gastrointestinal disorders
lipid modifying agents, plain
anti-parathyroid agents
ectoparasiticides, incl. scabicides
antipruritics, incl. antihistamines, anesthetics, etc.
antifungals for topical use
antiepileptics
cardiac glycosides
vasodilators used in cardiac diseases
nasal decongestants for systemic use
chemotherapeutics for topical use
antibiotics for topical use
adrenergics, inhalants
antiinflammatory and antirheumatic products, non-steroids
bile therapy
alkylating agents
other mineral supplements
other nervous system drugs
dopaminergic agents
anti-acne preparations for topical use
cardiac stimulants excl. cardiac glycosides
irrigating solutions
antihistamines for systemic use
hormone antagonists and related agents
urologicals
other analgesics and antipyretics
antipsychotics
muscle relaxants, centrally acting agents
vitamin b1, plain and in combination with vitamin b6 and b12
antimigraine preparations
angiotensin ii receptor blockers (arbs), plain
hormonal contraceptives for systemic use
antidepressants
hypothalamic hormones
corticosteroids, plain
calcium
potassium
intestinal antiinflammatory agents
propulsives
macrolides, lincosamides and streptogramins
arteriolar smooth muscle, agents acting on
muscle relaxants, peripherally acting agents
quinolone antibacterials
antiadrenergic agents, centrally acting
antifibrinolytics
high-ceiling diuretics
bacterial vaccines
opioids
glycogenolytic hormones
cough suppressants, excl. combinations with expectorants

Now, given the patient profile:
{task}
Please directly prescribe a appropriate set of drugs for the patient in the following format:
["drug1", "drug2", "drug3", ...]
"""
