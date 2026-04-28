PROBLEM_WITH_EVIDENCE_PROMPT = """{problem} Tips: {evidence}"""

CASE_PROMPT = """[Task] {task}
[Answer] ```sql\n{answer}\n```\n"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] ```sql\n{answer}\n```
[Feedback] {feedback}\n"""

CBR_PROMPT = """Given the schema of the database, your task is to answer the given question for the tables. Here are some relevant cases that may use same or different schemas:
{case_prompt}

Now, it's your turn. Here is the schema of the database:
{schema}

Using valid SQLite, generate the correct SQL code to answer the question below:
{task}
Please directly complete this task in the following code format:
```sql
-- Here is the SQL code.
```
"""


ZERO_SHOT_PROMPT = """Given the schema of the database, your task is to answer the given question for the tables. Here is the schema of the database:
{schema}

Now, using valid SQLite, generate the correct SQL code to answer the question below:
{task}
Please directly complete this task in the following code format:
```sql
-- Here is the SQL code.
```
"""
