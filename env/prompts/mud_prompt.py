CASE_PROMPT = """[Task] {task}
[Answer] ```json\n{answer}\n```\n"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] ```json\n{answer}\n```
[Feedback] {feedback}
"""

CBR_PROMPT = """Given the description of a legal case, your task is to decide what crime each defendant should be judged. Here are all the possible crimes with the format of "name of the crime: explanation" as below:
1. Offences of Illegal Manufacture of Firearms: The perpetrator violated state regulations on firearms management by illegally manufacturing firearms and endangering public safety.
2. Offences of Illegal Trade in Firearms: The perpetrator violated state regulations on firearms management by illegally trading in firearms and endangering public safety.
3. Offences of Illegal Possession of Firearms: Illegal possession of firearms without authorisation in violation of firearms regulations.
4. Offence of Selling Counterfeit Registered Goods: Selling goods that are known to be counterfeit registered trademarks and selling a large amount of them.
5. Contract Fraud: With the purpose of illegal possession, in the process of signing or fulfilling a contract, committing deceptive means such as fictitious facts or concealing the truth, to cheat the other party of property in a large amount.
6. Offence of Illegal Business Operation: Illegally engaging in business activities in violation of State regulations, disrupting the market order, under serious circumstances.
7. Offence of Counterfeiting a Registered Trademark: Violation of national trademark management regulations, without the permission of the owner of the registered trademark, in the same kind of goods and services, the use of the same trademark with its registered trademark, the circumstances are serious.
8. Intentional Homicide: Intentional and unlawful deprivation of life.
9. Intentional Injury: Acts of intentional unlawful damage to the physical integrity of another person.
10. Crime of Illegal Detention: Deliberate unlawful detention of a person or other unlawful deprivation of a person's personal liberty.
11. Robbery: Using violence, coercion or other methods to force the victim to hand over property on the spot, or forcibly snatching public or private property on the spot, for the purpose of unlawful appropriation.
12. Fraud: Fraudulently obtaining a larger amount of public or private property by means of fictitious facts or concealment of the truth for the purpose of unlawful appropriation.
13. Extortion and Blackmail: Intimidating, threatening or blackmailing the owner or manager of property for the purpose of unlawful appropriation, and forcibly soliciting a larger amount of public or private property.
14. Crime of Cheating and Bluffing: Fraudulent impersonation of the identity or title of a staff member of a State organ for the purpose of obtaining unlawful benefits, to the detriment of the prestige of the State organ and its normal activities.
15. Crime of Affray: Gathering of a number of persons to attack each other physically or to attack each other physically in order to disturb public order.
16. Crime of Picking Quarrels and Provoking Troubles: Acts of wanton provocation, randomly beating or harassing others or arbitrarily destroying or occupying public or private property, or acts of disturbances in public places that result in damages that seriously disrupt the social order.
17. Concealment of Proceeds of Crime: Concealing, transferring, acquiring, selling or otherwise disguising or concealing the proceeds of crime, knowing that they are proceeds of crime.
18. Harboring and Covering: Providing a place of concealment or property to a person who has committed a crime, knowing that he or she has done so, assisting him or her to escape or concealing his or her crime by means of false testimony.
19. Crime of Organisation of Prostitution: Recruiting, hiring, inducing, accommodating, etc., to gather and control a number of persons for the purpose of engaging in prostitution.
20. Crime of Facilitating the Organisation of Prostitution: Facilitating, creating conditions and removing obstacles for others to commit the offence of organising prostitution.
21. Crime of Harboring Prostitution: Provision of premises for the prostitution of others.
22. Crime of Procuring prostitution: Acts of matchmaking between persons engaged in prostitution and their clients.

Here are some relevant cases:
{case_prompt}

Now, given the description of a legal case:
[Task] {task}
Please answer this task in the following json format:
```json
{{"<Defendant A>": "The precise name of the crime for <Defendant A>", "<Defendant B>": "The precise name of the crime for <Defendant B>", ...}}
```
"""


ZERO_SHOT_PROMPT = """Given the description of a legal case, your task is to decide what crime each defendant should be judged. Here are all the possible crimes with the format of "name of the crime: explanation" as below:
1. Offences of Illegal Manufacture of Firearms: The perpetrator violated state regulations on firearms management by illegally manufacturing firearms and endangering public safety.
2. Offences of Illegal Trade in Firearms: The perpetrator violated state regulations on firearms management by illegally trading in firearms and endangering public safety.
3. Offences of Illegal Possession of Firearms: Illegal possession of firearms without authorisation in violation of firearms regulations.
4. Offence of Selling Counterfeit Registered Goods: Selling goods that are known to be counterfeit registered trademarks and selling a large amount of them.
5. Contract Fraud: With the purpose of illegal possession, in the process of signing or fulfilling a contract, committing deceptive means such as fictitious facts or concealing the truth, to cheat the other party of property in a large amount.
6. Offence of Illegal Business Operation: Illegally engaging in business activities in violation of State regulations, disrupting the market order, under serious circumstances.
7. Offence of Counterfeiting a Registered Trademark: Violation of national trademark management regulations, without the permission of the owner of the registered trademark, in the same kind of goods and services, the use of the same trademark with its registered trademark, the circumstances are serious.
8. Intentional Homicide: Intentional and unlawful deprivation of life.
9. Intentional Injury: Acts of intentional unlawful damage to the physical integrity of another person.
10. Crime of Illegal Detention: Deliberate unlawful detention of a person or other unlawful deprivation of a person's personal liberty.
11. Robbery: Using violence, coercion or other methods to force the victim to hand over property on the spot, or forcibly snatching public or private property on the spot, for the purpose of unlawful appropriation.
12. Fraud: Fraudulently obtaining a larger amount of public or private property by means of fictitious facts or concealment of the truth for the purpose of unlawful appropriation.
13. Extortion and Blackmail: Intimidating, threatening or blackmailing the owner or manager of property for the purpose of unlawful appropriation, and forcibly soliciting a larger amount of public or private property.
14. Crime of Cheating and Bluffing: Fraudulent impersonation of the identity or title of a staff member of a State organ for the purpose of obtaining unlawful benefits, to the detriment of the prestige of the State organ and its normal activities.
15. Crime of Affray: Gathering of a number of persons to attack each other physically or to attack each other physically in order to disturb public order.
16. Crime of Picking Quarrels and Provoking Troubles: Acts of wanton provocation, randomly beating or harassing others or arbitrarily destroying or occupying public or private property, or acts of disturbances in public places that result in damages that seriously disrupt the social order.
17. Concealment of Proceeds of Crime: Concealing, transferring, acquiring, selling or otherwise disguising or concealing the proceeds of crime, knowing that they are proceeds of crime.
18. Harboring and Covering: Providing a place of concealment or property to a person who has committed a crime, knowing that he or she has done so, assisting him or her to escape or concealing his or her crime by means of false testimony.
19. Crime of Organisation of Prostitution: Recruiting, hiring, inducing, accommodating, etc., to gather and control a number of persons for the purpose of engaging in prostitution.
20. Crime of Facilitating the Organisation of Prostitution: Facilitating, creating conditions and removing obstacles for others to commit the offence of organising prostitution.
21. Crime of Harboring Prostitution: Provision of premises for the prostitution of others.
22. Crime of Procuring prostitution: Acts of matchmaking between persons engaged in prostitution and their clients.

Now, given the description of a legal case:
{task}
Please answer this task in the following json format:
```json
{{"<Defendant A>": "The precise name of the crime for <Defendant A>", "<Defendant B>": "The precise name of the crime for <Defendant B>", ...}}
```
"""
