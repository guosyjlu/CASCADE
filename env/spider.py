"""
This is the environment file for SPIDER.
- Code generation: text-to-SQL.
[Ref.1] Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task
[Ref.2] StreamBench: Towards Benchmarking Continuous Improvement of Language Agents
"""
import os
import sys
import json
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
from .base import Env
from .prompts.spider_prompt import ZERO_SHOT_PROMPT, CASE_PROMPT, CBR_PROMPT, CASE_PROMPT_REFLEXION

def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

def execute_model(predicted_sql, ground_truth, db_path, meta_time_out=30):
    try:
        res = func_timeout(
            meta_time_out, 
            execute_sql,
            args=(predicted_sql, ground_truth, db_path)
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
    except Exception as e:
        res = 0
    return res

def nice_look_table(column_names: list, values: list):
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    full_schema_prompt_list = []
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            if table == 'sqlite_sequence':
                continue
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            create_prompt = cursor.fetchone()[0]
            schemas[table[0]] = create_prompt
            if num_rows:
                cur_table = table[0]
                if cur_table in ['order', 'by', 'group']:
                    cur_table = "`{}`".format(cur_table)

                cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
                column_names = [description[0] for description in cursor.description]
                values = cursor.fetchall()
                rows_prompt = nice_look_table(column_names=column_names, values=values)
                verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
                schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for v in schemas.values():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt


class SPIDEREnv(Env):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.db_path = "data/spider/source/"
        self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CASE_PROMPT_REFLEXION = CASE_PROMPT_REFLEXION
        self.CBR_PROMPT = CBR_PROMPT
        
    def init_env(self):
        # Load dataset
        with open("data/spider/spider.jsonl", encoding="utf-8") as file:
            for line in file:
                sample = json.loads(line)
                self.dataset.append(sample)
        # Load schemas
        self.db_prompt_schema = dict()
        for row in self.dataset:
            if row["db_id"] not in self.db_prompt_schema:
                db_path = os.path.join(self.db_path, row["db_id"], row["db_id"] + ".sqlite")
                schema_prompt = generate_schema_prompt(db_path)
                self.db_prompt_schema[row["db_id"]] = schema_prompt
                
    def observe(self):
        problem = self.dataset[self.index]["task"]
        return problem
    
    def evaluate(self, generated_text):
        generated_answer = self.extraction(generated_text)
        reward = self.reward_function(generated_answer)
        return generated_answer, reward
    
    def get_zero_shot_prompt(self, problem):
        assert self.observe() == problem
        db_id = self.dataset[self.index]["db_id"]
        schema = self.db_prompt_schema[db_id]
        return self.ZERO_SHOT_PROMPT.format(task=problem, schema=schema)
    
    def get_case_based_prompt(self, problem, cases):
        assert self.observe() == problem
        db_id = self.dataset[self.index]["db_id"]
        schema = self.db_prompt_schema[db_id]
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem, schema=schema)
    
    def get_case_based_prompt_reflexion(self, problem, cases):
        assert self.observe() == problem
        db_id = self.dataset[self.index]["db_id"]
        schema = self.db_prompt_schema[db_id]
        case_prompt = ""
        for case in cases:
            if case["reward"] == 1:
                feedback = "This answer is correct!"
            else:
                feedback = "This answer is wrong!"
            case_prompt += self.CASE_PROMPT_REFLEXION.format(task=case["task"], answer=case["answer"], feedback=feedback)
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem, schema=schema)
    
    def __len__(self):
        return len(self.dataset)
    
    def extraction(self, generated_text):
        try:
            answer = generated_text.split('```sql')[-1].split('```')[0].strip()
        except Exception:
            answer = None
        return answer

    def reward_function(self, generated_answer):
        if generated_answer is None:
            reward = 0
        else:
            ground_truth_script = self.dataset[self.index]["query"]
            db_id = self.dataset[self.index]["db_id"]
            db_path = os.path.join(self.db_path, db_id, db_id+'.sqlite')
            reward = execute_model(generated_answer, ground_truth_script, db_path)
        return reward

    def get_ground_truth(self):
        return self.dataset[self.index]["query"]