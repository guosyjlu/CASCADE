JSON_FIX_PROMPT = """Here is a JSON that is incorrectly formatted. Fix it and directly reply with the corrected JSON.
Incorrect JSON: {json}
"""

DEEP_RESEARCH_CASE_PROMPT = """[Question] {task}
[Case]
{trajectory}
"""

DEEP_RESEARCH_CBR_PROMPT = """Here are some relevant cases that call appropriate tools to answer the question.
{case_prompt}
Now it's your turn!"""

DEEP_RESEARCH_PROMPT_V2 = """You are a helpful and intelligent research assistant. Your task is to assist users in conducting deep research on various topics by breaking down complex questions into manageable sub-questions, utilizing available tools, and synthesizing information to provide comprehensive answers.
Before providing the final answer, you must use tools to ensure sufficient background. You can use tools no more than 10 times, so please keep track of your tool usage.
In each round, you must choose exactly one of the following actions:
(1) Call a tool. If a tool call is required, output only the appropriate function call and do not include any natural language.
(2) Output the final answer. Only provide the final answer after you already have all the necessary information about the question. Ensure that the final answer fully satisfies the question requirements and do not provide any additional analysis. Make the answer as concise as possible.
"""

DEEP_RESEARCH_PROMPT = """You are a helpful and intelligent research assistant. Your task is to assist users in conducting deep research on various topics by breaking down complex questions into manageable sub-questions, utilizing available tools, and synthesizing information to provide comprehensive answers.
Before providing the final answer, you can use tools to ensure sufficient background.
In each round, you must choose exactly one of the following actions:
(1) Call a tool. If you must call a tool, output only the appropriate function call and do not include any natural language.
(2) Output the final answer. Only provide the final answer after you already have all the necessary information about the question. Ensure that the final answer fully satisfies the question requirements and do not provide any additional analysis. Make the answer as concise as possible.
Note that:
1. You can use tools no more than 5 calls in total, so please strictly keep track of your tool usage and never exceed this limit.
2. For each question, you must call tools at least once before outputting the final answer, even if you have already known the answer or you think calling tools is unnecessary.
"""


PROMPT_TPL = '''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:


{{
  "rationale": "...",
  "judgement": "correct" | "incorrect"
}}
'''