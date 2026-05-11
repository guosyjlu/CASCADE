# CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment

<p align="center">
<a href="https://huggingface.co/datasets/guosy/DTLBench">
🤗 DTLBench
</a> |
<a href="https://physionet.org/content/mimic-iv-ext-dtlbench/1.0.0/">
🫀 DTLBench (To be released)
</a> |
<a href="https://arxiv.org/abs/2605.06702">
📄 Paper
</a>
</p>

<p align="center">
<strong>English</strong> | <a href="README_zh.MD">中文</a>
</p>

<p align="center">
<a href="#whats-new">What's New</a> |
<a href="#quick-start">Quick Start</a> |
<a href="#resource-download">Resource Download</a> |
<a href="#experiment-reproduction">Experiment Reproduction</a> |
<a href="#customized-environment-for-deployment-time-learning-via-cascade">Customized Environment</a> |
<a href="#code-structure">Code Structure</a>
<a href="#citation">Citation</a>
</p>

<table>
<tr><td>
<p align="center">
  <img src="Figures/Figure-1.png" width="100%" alt="LLM Lifecycle">
</p>
<p align="left"><sub><b>The LLM Lifecycle.</b> In the first stage, LLMs are pretrained with next-token prediction tasks on a large scale of corpus. Then, LLMs are further finetuned using SFT and RLFT for alignment and enhancing reasoning capabilities. We consider deployment-time learning as the third stage, where LLMs learn from experience during deployment, enabling continuous policy improvement over online interactions without updating the underlying LLM parameters.</sub></p>
</td></tr>
</table>

<table>
<tr><td>
<p align="center">
  <img src="Figures/Figure-2.png" width="100%" alt="CASCADE">
</p>
<p align="left"><sub><b>Overview of CASCADE.</b> Given a query, CASCADE retrieves the case via the contextual bandit algorithm, reuses and revises it to generate the solution, and receives the reward. The retriever policy is updated accordingly, and successful cases are retained in the case bank.</sub></p>
</td></tr>
</table>

<a id="whats-new"></a>
## What's New

- `2026-05-11`: CASCADE paper is available via arXiv.
- `2026-04-28`: CASCADE is open-sourced.

<a id="quick-start"></a>
## Quick Start

The minimal steps for running CASCADE on DTLBench are:

1. Set up the Python environment
2. Download DTLBench
3. Run `main.py` with a supported environment and model backend

### 1. Set Up the Environment

```bash
cd CASCADE
conda create -n cascade python=3.10
conda activate cascade

# install torch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Optional: install flash-attn (only required by baseline REINFORCE+LoRA)
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt
```

### 2. Download DTLBench

For open-sourced datasets, you can manually download them from <a href="https://huggingface.co/datasets/guosy/DTLBench">
🤗 Huggingface datasets
</a> or use:

```bash
cd CASCADE
mkdir -p data
huggingface-cli download --repo-type dataset --resume-download guosy/DTLBench --local-dir data
```

For restricted-license datasets, please manually download them from PhysioNet after completing the required training. **NOTE:** The related datasets will be open-sourced after publication due to the policy from PhysioNet.

### 3. Run CASCADE

A minimal `vllm` example (via local deployment) is:

```bash
python main.py \
  --seed 0 \
  --env ddxplus \
  --agent cbr \
  --bandit NeuralLinLogUCB \
  --llm qwen3-32b \
  --serving_mode vllm \
  --server <YOUR_vLLM_SERVER_IP> \
  --port <YOUR_vLLM_PORT> \
  --learning_rate 1e-5 \
  --nu 0.1
```

A minimal `openai` example (via OpenAI compatible API) is:

```bash
python main.py \
  --seed 0 \
  --env ddxplus \
  --agent cbr \
  --bandit NeuralLinLogUCB \
  --llm gemini-2.0-flash \
  --serving_mode openai \
  --learning_rate 1e-5 \
  --nu 0.1
```

The most important arguments are:

- `--env`: the DTLBench task to run, such as `ddxplus`, `spider`, `bird`, `banking77`, or `sentifin` (as specified in [env/__init__.py](env/__init__.py))
- `--agent`: the deployment-time learning method; use `cbr` for CASCADE
- `--bandit`: the contextual bandit algorithm used to rank recalled cases; use `NeuralLinLogUCB` for CASCADE
- `--llm`: the LLM name passed to the backend
- `--serving_mode`: choose `openai` for API-compatible serving or `vllm` for local serving
- `--server`: the host name of the vLLM server
- `--port`: the port of the vLLM server
- `--learning_rate`: learning rate for training reward model
- `--nu`: exploration coefficient in contextual bandit algorithms

<a id="resource-download"></a>
## Resource Download

This section is only required for specific tasks. If you run tasks such as `ddxplus`, `banking77`, or `sentifin`, you do not need the extra resources below.

Some DTLBench tasks require additional resource files:

1. **BIRD**: Download the resource files from [https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) and save them to `data/bird/source/`. Then unzip the archive and recursively extract any nested zip files until no zip files remain.

2. **SPIDER**: Download the resource files from [https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing) and save them to `data/spider/source/`. Then unzip the archive, and only keep `test_tables.json`, `test.json`, and all the folders in `test_database` in `data/spider/source`.

3. **MIMIC-III (EHR)**: Download the resource files from [https://drive.google.com/file/d/1diCy549_IM-iXmXdhLEiDfv-X44-2ewz/view?usp=sharing](https://drive.google.com/file/d/1diCy549_IM-iXmXdhLEiDfv-X44-2ewz/view?usp=sharing) and save them to `data/ehr/source/`. Then unzip the archive.

4. **2wiki**: Download the sources using the provided script. Note that the files are large, so the download may take a significant amount of time.

```bash
cd data/2wiki
bash download.sh ./source
```

<a id="experiment-reproduction"></a>
## Experiment Reproduction
We have provided scripts for all the main experiments in the paper.

For local deployment of Qwen models, please refer to files (`serve_qwen_4b.sh`, `serve_qwen_8b.sh`, `serve_qwen_14b.sh`, `serve_qwen_32b.sh`, `serve_qwen_30b.sh`) in the `scripts` directory.

For single-turn results in Fig. 3, please refer to the file `single_turn_experiments.sh` in the `scripts` directory.

For multi-turn, simulated results in Fig. 5, please refer to the file `multi_turn_experiments.sh` in the `scripts` directory.

For multi-turn, real-world results in Fig. 6, please refer to the file `real_world_experiments.sh` in the `scripts` directory.

For LLM scalability results in Fig.4, please refer to the file `white_box_experiments.sh` and `black_box_experiments.sh` in the `scripts` directory.

<a id="customized-environment-for-deployment-time-learning-via-cascade"></a>
## Customized Environment for Deployment-Time Learning via CASCADE

To plug a new task into CASCADE, the simplest path is to add a new single-turn environment by following [env/base.py](env/base.py) and an existing example such as [env/ddxplus.py](env/ddxplus.py) or [env/spider.py](env/spider.py). For CASCADE, the natural implementation order is:

1. create `env/<task>.py`
2. create `env/prompts/<task>_prompt.py`
3. register the environment in `env/__init__.py`
4. run `main.py`

<details>
<summary><strong>1. Configure <code>env/&lt;task&gt;.py</code></strong></summary>

This file is the core of a single-turn environment. It should subclass `Env` and implement the methods that CASCADE calls during deployment-time learning:

- `__len__()`: number of samples
- `observe()`: return the current task text
- `evaluate(generated_text)`: parse model output and return `(generated_answer, reward)`
- `get_zero_shot_prompt(problem)`: build the prompt when no case is available
- `get_case_based_prompt(problem, cases)`: build the prompt when CASCADE retrieves historical cases

In this file, you should complete three things:

- load your task stream in `init_env()`
- define how to extract the final answer from the model output
- define how to compute the reward

What each sample should expose is more important than the storage format. In most tasks, a sample should provide at least:

- `task`: the exact text used for prompting and retrieval
- ground-truth supervision for evaluation, often stored as `label`

If your task needs more context, such as schema, label space, API docs, or business rules, just store them in each sample and inject them into the prompt when needed.

A minimal environment looks like this:

```python
import json
from .base import Env
from .prompts.my_task_prompt import ZERO_SHOT_PROMPT, CASE_PROMPT, CBR_PROMPT

class MyTaskEnv(Env):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.init_env()
        self.ZERO_SHOT_PROMPT = ZERO_SHOT_PROMPT
        self.CASE_PROMPT = CASE_PROMPT
        self.CBR_PROMPT = CBR_PROMPT

    def init_env(self):
        with open("data/my_task/my_task.jsonl", encoding="utf-8") as file:
            for line in file:
                self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def observe(self):
        return self.dataset[self.index]["task"]

    def evaluate(self, generated_text):
        generated_answer = self.extraction(generated_text)
        reward = self.reward_function(generated_answer)
        return generated_answer, reward

    def get_zero_shot_prompt(self, problem):
        return self.ZERO_SHOT_PROMPT.format(task=problem)

    def get_case_based_prompt(self, problem, cases):
        case_prompt = ""
        for case in cases:
            case_prompt += self.CASE_PROMPT.format(task=case["task"], answer=case["answer"])
        return self.CBR_PROMPT.format(case_prompt=case_prompt, task=problem)

    def extraction(self, generated_text):
        return generated_text.strip()

    def reward_function(self, generated_answer):
        ground_truth = self.dataset[self.index]["label"]
        return int(generated_answer == ground_truth)
```

The key point in `evaluate()` is that it should contain both steps CASCADE needs:

- `extraction(generated_text)`: extract the final answer from the raw completion
- `reward_function(generated_answer)`: compare it with ground truth and return `0` or `1`

</details>

<details>
<summary><strong>2. Configure <code>env/prompts/&lt;task&gt;_prompt.py</code></strong></summary>

For CASCADE, the prompt file only needs three templates:

- `ZERO_SHOT_PROMPT`
- `CASE_PROMPT`
- `CBR_PROMPT`

Their roles are:

- `ZERO_SHOT_PROMPT`: used when there is no retained case yet
- `CASE_PROMPT`: defines how one historical case is serialized
- `CBR_PROMPT`: wraps retrieved cases and the current task into the final prompt

These prompts should contain the following required information:

1. Task instruction: what the model is supposed to do.
2. Task-specific context: anything the model needs but is not already in `task`.
3. Output format: a strict answer format that can be parsed reliably.
4. Case consistency: the answer format in `CASE_PROMPT` must be the same format expected by the current task.

A minimal example is:

```python
CASE_PROMPT = """[Task] {task}
[Answer] {answer}
"""

CBR_PROMPT = """You are a helpful assistant for my task.

Here are some relevant cases:
{case_prompt}

Now solve the following task:
{task}

Please output the answer in the format:
<answer>
"""

ZERO_SHOT_PROMPT = """You are a helpful assistant for my task.

Now solve the following task:
{task}

Please output the answer in the format:
<answer>
"""
```

The key rule is that prompt format and evaluation format must match. If the prompt asks for `\\boxed{answer}`, then `extraction()` should parse `\\boxed{...}`. If the task is SQL generation, the prompt should force a single SQL block, and evaluation should check execution results instead of raw string equality.

</details>

<details>
<summary><strong>3. Register in <code>env/__init__.py</code></strong></summary>

After implementing the environment and prompt file, register the new environment in [env/__init__.py](env/__init__.py):

```python
from .my_task import MyTaskEnv

ENV_DICT = {
    # ...
    "my-task": MyTaskEnv,
}
```

</details>

<details>
<summary><strong>4. Run CASCADE</strong></summary>

and run:

```bash
python main.py --env my-task --agent cbr --bandit NeuralLinLogUCB --llm <model_name> --serving_mode openai
```

</details>

<details>
<summary><strong>Checklist</strong></summary>

Before running, check these five items:

1. `task` is the exact text you want CASCADE to retrieve on.
2. `ZERO_SHOT_PROMPT` and `CBR_PROMPT` ask for the same answer format.
3. `CASE_PROMPT` stores answers in the same format expected at inference time.
4. `extraction()` can robustly parse that format.
5. `reward_function()` reflects the real task objective.

If these pieces are correct, a new single-turn task can usually be integrated into CASCADE with very little additional work.

</details>

<a id="code-structure"></a>
## Code Structure

```text
CASCADE/
├─ main.py                 # Main entry: most experiments are conducted via this script
├─ main_discovery.py       # Experiments for discovery mechanism in the supplementary notes
├─ main_deepsearch.py      # Experiments for deep search (required MCP tools)
├─ agent.py                # Agent implementation
├─ bandit.py               # Bandit policy implementation
├─ config.py               # Unified configuration
├─ llm.py                  # OpenAI / vLLM interface for calling LLMs
├─ data/                   # Directory for datasets and resourced in DTLBench
├─ env/                    # Environment class and prompts for all the tasks in DTLBench
├─ scripts/                # scripts for vLLM deployment and experiments
├─ Figures/
└─ requirements.txt
```

<a id="citation"></a>
## Citation

Please consider citing our paper if you find it useful.
```
@misc{guo2026cascadecasebasedcontinualadaptation,
      title={CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment}, 
      author={Siyuan Guo and Yali Du and Hechang Chen and Yi Chang and Jun Wang},
      year={2026},
      eprint={2605.06702},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.06702}, 
}
```
