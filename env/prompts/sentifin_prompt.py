CASE_PROMPT = """[Task] {task}
[Answer] {answer}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] {answer}
[Feedback] {feedback}
"""

CBR_PROMPT = """Given a piece of user query, your task is to identify the entities which are **companies** or **organizations** from the content, and then classify the sentiment of the corresponding entities into neutral, positive or negative. 
Here are some relevant cases:
{case_prompt}

Now, given the user query:
{task}
Note that some entities may appear multiple times. Please provide the entity-sentiment pairs in order for each occurrence.
Now, please directly list all the potential entity-sentiment pairs exactly with the following format. Do not output any other redundant words.
{{'entity': entity_name1, 'label': sentiment1 (neutral, positive, or negative)}}
{{'entity': entity_name2, 'label': sentiment2 (neutral, positive, or negative)}}
...
"""


ZERO_SHOT_PROMPT = """Given a piece of user query, your task is to identify the entities which are **companies** or **organizations** from the content, and then classify the sentiment of the corresponding entities into neutral, positive or negative. 
Now, given the user query:
{task}
Note that some entities may appear multiple times. Please provide the entity-sentiment pairs in order for each occurrence.
Now, please directly list all the potential entity-sentiment pairs exactly with the following format. Do not output any other redundant words.
{{'entity': entity_name1, 'label': sentiment1 (neutral, positive, or negative)}}
{{'entity': entity_name2, 'label': sentiment2 (neutral, positive, or negative)}}
...
"""
