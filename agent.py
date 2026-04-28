import copy
import torch
import random
import numpy as np
from collections import deque
from bandit import PretrainedBandit, NeuralLinLogUCB, LinearLogisticUCB, NeuralLogUCB, NeuralLinUCB
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class ZeroShotAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
    
    def solve(self, problem):
        prompt = self.env.get_zero_shot_prompt(problem)
        return prompt

    def learn(self, problem, generated_answer, reward):
        return {}

class ICRLAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.case_bank = []

    def solve(self, problem):
        # 1. Get latest k cases
        self.retrieved_cases = self.case_bank[-self.args.comk:]
        # 2. ICRL
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt_reflexion(problem, self.retrieved_cases)
        return prompt

    def learn(self, problem, generated_answer, reward):
        # Retain
        new_case = {"task": problem, "answer": generated_answer, "reward": reward}
        self.case_bank.append(new_case)
        return {}

class ICRLPlusAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.case_bank = []

    def solve(self, problem):
        # 1. Get latest k cases
        self.retrieved_cases = self.case_bank[-self.args.comk:]
        # 2. ICRL
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt(problem, self.retrieved_cases)
        return prompt

    def learn(self, problem, generated_answer, reward):
        # Retain
        if reward == 1:
            new_case = {"task": problem, "answer": generated_answer}
            self.case_bank.append(new_case)
        return {}

class CBRAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.model = AutoModel.from_pretrained(args.embedding_model_path).to("cuda")
        self.model.eval()
        
        # Load bandit object
        if args.bandit == "Pretrained":
            self.bandit = PretrainedBandit(args)
        elif args.bandit == "LinLogUCB":
            self.bandit = LinearLogisticUCB(args)
        elif args.bandit == "NeuralLogUCB":
            self.bandit = NeuralLogUCB(args)
        elif args.bandit == "NeuralLinUCB":
            self.bandit = NeuralLinUCB(args)
        elif args.bandit == "NeuralLinLogUCB":
            self.bandit = NeuralLinLogUCB(args)    
        else:
            raise NotImplementedError("The specified bandit is not supported yet.")
            
        self.case_bank = []
        self.case_embeddings = None
        self.metrics = dict()

    def solve(self, problem):
        # 1. Retrieve
        self.retrieved_cases = self.retrieve(problem)
        # 2. Reuse & Revise
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt(problem, self.retrieved_cases)
        return prompt

    def learn(self, problem, generated_answer, reward):
        if reward == 1:
            new_case = {"task": problem, "answer": generated_answer}
            self.case_bank.append(new_case)
            embedding = self.encode(problem).reshape(1, -1)
            if self.case_embeddings is None:
                self.case_embeddings = embedding
            else:
                self.case_embeddings = torch.cat((self.case_embeddings, embedding), dim=0)
                
        # Bandit Learning
        if len(self.retrieved_cases) != 0:
            metrics = self.bandit.learn(problem, self.retrieved_cases, reward)
            self.metrics.update(metrics)
        
        # Update metric
        metrics = copy.deepcopy(self.metrics)
        self.metrics = {}
        return metrics
    
    @torch.no_grad()
    def encode(self, problem):
        inputs = self.tokenizer([problem], return_tensors='pt').to("cuda")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # L2 normalization
        return embeddings
    
    @torch.no_grad()
    def recall(self, problem):
        q_embedding = self.encode(problem).reshape(1, -1)
        similarity = q_embedding@self.case_embeddings.T
        similarity = similarity.reshape(-1)
        k = min(len(self.case_bank), self.args.recall_size)
        _, index = torch.topk(similarity, k)
        return [self.case_bank[i] for i in index]
        
    def retrieve(self, problem):
        if len(self.case_bank) <= self.args.comk:
            retrieved_cases = [case for case in self.case_bank]
        else:
            # Recall via pretrained embedding model
            recalled_cases = self.recall(problem)
            selected_index, metrics = self.bandit.rank(problem, recalled_cases, self.args.comk)
            retrieved_cases = [recalled_cases[i] for i in selected_index]
            self.metrics.update(metrics)
        return retrieved_cases


class REINFORCEAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True, 
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True, 
            attn_implementation="flash_attention_2"
        )
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        self.device = self.model.device
        self.accumulated_steps = 0
        self.total_loss = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
    
    def solve(self, problem):
        prompt = self.env.get_zero_shot_prompt(problem)
        return prompt
    
    def generate_response(self, prompt):
        """
        This generation follows Qwen3 (non-thinking). You can customize this function as needed.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        template_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(template_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                top_k=20,
                temperature=0.1,
                top_p=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        # Keep data logs in track
        self.gen_data = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "generated_ids": generated_ids
        }
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    def compute_log_probs(self, input_ids, generated_ids, use_lora=True):
        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        if not use_lora:
            self.model.disable_adapter_layers()
        with torch.set_grad_enabled(use_lora):
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids)
            )
            logits = outputs.logits
        if not use_lora:
            self.model.enable_adapter_layers()
        
        gen_logits = logits[:, input_ids.shape[1]-1:-1, :]  # Align with generated_ids
        log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, 
            index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)
        sequence_log_prob = token_log_probs.sum(dim=-1)
        
        return sequence_log_prob

    def compute_kl_divergence(self, input_ids, generated_ids):
        policy_log_prob = self.compute_log_probs(input_ids, generated_ids, use_lora=True)
        with torch.no_grad():
            ref_log_prob = self.compute_log_probs(input_ids, generated_ids, use_lora=False)
        kl = policy_log_prob - ref_log_prob
        return kl

    def learn(self, problem, generated_answer, reward):
        # Special Case: MIMIC-IV (We explose ground-truth in this setting)
        if self.args.env == "mimiciv" and reward == 1:
            generated_text = str(generated_answer) + self.tokenizer.eos_token
            generated_ids = self.tokenizer.encode(generated_text, return_tensors="pt").to(self.device)
            self.gen_data["generated_ids"] = generated_ids
        
        policy_log_prob = self.compute_log_probs(
            self.gen_data["input_ids"], 
            self.gen_data["generated_ids"],
            use_lora=True
        )
        kl = self.compute_kl_divergence(self.gen_data["input_ids"], self.gen_data["generated_ids"])
        loss = -(reward * policy_log_prob - self.args.kl_coef * kl)
        loss = loss / self.args.batch_size
        self.total_loss += loss.item()
        loss.backward()
        self.accumulated_steps += 1
        
        metrics = {
            "Training/polcy_log_prob": policy_log_prob.item(),
            "Training/kl": kl.item(),
            "Training/loss": loss.item(),
        }
        
        if self.accumulated_steps % self.args.batch_size == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            metrics["Training/batch_loss"] = self.total_loss
            print(f"Iter-{self.accumulated_steps}, Loss={self.total_loss}")
            self.total_loss = 0
        
        return metrics
    
    def cleanup(self):
        try:
            self.model.to("cpu")
            del self.model
            del self.base_model
            del self.optimizer
            del self.tokenizer
        except Exception as e:
            print(f"Cleanup error: {e}")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class CBRGTAgent:
    """
    CBR with ground-truth label during Retain step.
    """
    def __init__(self, args, env):
        self.args = args
        self.env = env
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.model = AutoModel.from_pretrained(args.embedding_model_path).to("cuda")
        self.model.eval()
        
        # Load bandit object
        if args.bandit == "Pretrained":
            self.bandit = PretrainedBandit(args)
        elif args.bandit == "LinLogUCB":
            self.bandit = LinearLogisticUCB(args)
        elif args.bandit == "NeuralLogUCB":
            self.bandit = NeuralLogUCB(args)
        elif args.bandit == "NeuralLinUCB":
            self.bandit = NeuralLinUCB(args)
        elif args.bandit == "NeuralLinLogUCB":
            self.bandit = NeuralLinLogUCB(args)    
        else:
            raise NotImplementedError("The specified bandit is not supported yet.")
            
        self.case_bank = []
        self.case_embeddings = None
        self.metrics = dict()

    def solve(self, problem):
        # 1. Retrieve
        self.retrieved_cases = self.retrieve(problem)
        # 2. Reuse & Revise
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt(problem, self.retrieved_cases)
        return prompt

    def learn(self, problem, generated_answer, reward):
        # Retain with Ground-Truth
        ground_truth = self.env.get_ground_truth()
        new_case = {"task": problem, "answer": ground_truth}
        self.case_bank.append(new_case)
        embedding = self.encode(problem).reshape(1, -1)
        if self.case_embeddings is None:
            self.case_embeddings = embedding
        else:
            self.case_embeddings = torch.cat((self.case_embeddings, embedding), dim=0)
                
        # Bandit Learning
        if len(self.retrieved_cases) != 0:
            metrics = self.bandit.learn(problem, self.retrieved_cases, reward)
            self.metrics.update(metrics)
        
        # Update metric
        metrics = copy.deepcopy(self.metrics)
        self.metrics = {}
        return metrics
    
    @torch.no_grad()
    def encode(self, problem):
        inputs = self.tokenizer([problem], return_tensors='pt').to("cuda")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # L2 normalization
        return embeddings
    
    @torch.no_grad()
    def recall(self, problem):
        q_embedding = self.encode(problem).reshape(1, -1)
        similarity = q_embedding@self.case_embeddings.T
        similarity = similarity.reshape(-1)
        k = min(len(self.case_bank), self.args.recall_size)
        _, index = torch.topk(similarity, k)
        return [self.case_bank[i] for i in index]
        
    def retrieve(self, problem):
        if len(self.case_bank) <= self.args.comk:
            retrieved_cases = [case for case in self.case_bank]
        else:
            # Recall via pretrained embedding model
            recalled_cases = self.recall(problem)
            selected_index, metrics = self.bandit.rank(problem, recalled_cases, self.args.comk)
            retrieved_cases = [recalled_cases[i] for i in selected_index]
            self.metrics.update(metrics)
        return retrieved_cases



class CBRDiscoveryAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.model = AutoModel.from_pretrained(args.embedding_model_path).to("cuda")
        self.model.eval()
        
        # Load bandit object
        if args.bandit == "Pretrained":
            self.bandit = PretrainedBandit(args)
        elif args.bandit == "LinLogUCB":
            self.bandit = LinearLogisticUCB(args)
        elif args.bandit == "NeuralLogUCB":
            self.bandit = NeuralLogUCB(args)
        elif args.bandit == "NeuralLinUCB":
            self.bandit = NeuralLinUCB(args)
        elif args.bandit == "NeuralLinLogUCB":
            self.bandit = NeuralLinLogUCB(args)    
        else:
            raise NotImplementedError("The specified bandit is not supported yet.")
            
        self.case_bank = []
        self.case_embeddings = None
        self.metrics = dict()
        
        # Discovery variables
        self.index = 0
        self.costs = 0
        self.queue = deque(maxlen=args.queue_size)
        self.budget = int(self.args.budget_ratio * len(self.env.dataset))
        self.sample_prob = self.budget / (len(self.env.dataset) - self.args.skip_steps)

    def solve(self, problem):
        # 1. Retrieve
        isDiscovery, self.retrieved_cases = self.retrieve(problem)
        # 2. Reuse & Revise
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt(problem, self.retrieved_cases)
        return isDiscovery, prompt

    def learn(self, problem, generated_answer, reward, isDiscovery):
        if reward == 1 or isDiscovery:
            if isDiscovery:
                new_case = {"task": problem, "answer": self.env.get_ground_truth()}
            else:
                new_case = {"task": problem, "answer": generated_answer}
            self.case_bank.append(new_case)
            embedding = self.encode(problem).reshape(1, -1)
            if self.case_embeddings is None:
                self.case_embeddings = embedding
            else:
                self.case_embeddings = torch.cat((self.case_embeddings, embedding), dim=0)
                
        # Bandit Learning
        if len(self.retrieved_cases) != 0:
            metrics = self.bandit.learn(problem, self.retrieved_cases, reward)
            self.metrics.update(metrics)
        
        # Update metric
        metrics = copy.deepcopy(self.metrics)
        self.metrics = {}
        return metrics
    
    @torch.no_grad()
    def encode(self, problem):
        inputs = self.tokenizer([problem], return_tensors='pt').to("cuda")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # L2 normalization
        return embeddings
    
    @torch.no_grad()
    def recall(self, problem):
        q_embedding = self.encode(problem).reshape(1, -1)
        similarity = q_embedding@self.case_embeddings.T
        similarity = similarity.reshape(-1)
        k = min(len(self.case_bank), self.args.recall_size)
        _, index = torch.topk(similarity, k)
        return [self.case_bank[i] for i in index]
    
    def discovery(self):
        # If the discovery costs > budget
        if self.costs >= self.budget:
            return False
        
        # Handling two types of discovery strategies
        if self.args.discover_strategy == "random":
            if self.index <= self.args.skip_steps:
                self.index += 1
                return False
            if random.random() < self.sample_prob:
                self.costs += 1
                return True
            else:
                return False
        else:
            discovery_metric = self.metrics.get(f"UCB/{self.args.discover_strategy}_max", None)
            if discovery_metric is None:
                return False
            
            if self.index < self.args.skip_steps:
                self.queue.append(discovery_metric)
                self.index += 1
                return False
            else:
                threshold = np.percentile(self.queue, 100*self.sample_prob)
                if discovery_metric < threshold:
                    self.queue.append(discovery_metric)
                    self.costs += 1
                    return True
                else:
                    self.queue.append(discovery_metric)
                    return False
        
    def retrieve(self, problem):
        if len(self.case_bank) <= self.args.comk:
            retrieved_cases = [case for case in self.case_bank]
        else:
            # Recall via pretrained embedding model
            recalled_cases = self.recall(problem)
            selected_index, metrics = self.bandit.rank(problem, recalled_cases, self.args.comk)
            retrieved_cases = [recalled_cases[i] for i in selected_index]
            self.metrics.update(metrics)
        isDiscovery = self.discovery()
        return isDiscovery, retrieved_cases


class DeepResearchZeroShotAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
    
    def solve(self, problem):
        prompt = self.env.get_zero_shot_prompt(problem)
        return prompt

    def learn(self, problem, summary, trajectory, reward):
        return {}

class DeepResearchCBRAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.model = AutoModel.from_pretrained(args.embedding_model_path).to("cuda")
        self.model.eval()
        
        # Load bandit object
        if args.bandit == "Pretrained":
            self.bandit = PretrainedBandit(args)
        elif args.bandit == "LinLogUCB":
            self.bandit = LinearLogisticUCB(args)
        elif args.bandit == "NeuralLogUCB":
            self.bandit = NeuralLogUCB(args)
        elif args.bandit == "NeuralLinUCB":
            self.bandit = NeuralLinUCB(args)
        elif args.bandit == "NeuralLinLogUCB":
            self.bandit = NeuralLinLogUCB(args)    
        else:
            raise NotImplementedError("The specified bandit is not supported yet.")
            
        self.case_bank = []
        self.case_embeddings = None
        self.metrics = dict()

    def solve(self, problem):
        # 1. Retrieve
        self.retrieved_cases = self.retrieve(problem)
        # 2. Reuse & Revise
        if len(self.retrieved_cases) == 0:
            # Zero-shot
            prompt = self.env.get_zero_shot_prompt(problem)
        else:
            # Case-based
            prompt = self.env.get_case_based_prompt(problem, self.retrieved_cases)
        return prompt

    def learn(self, problem, summary, trajectory, reward):
        if reward == 1 and summary != "":
            task = f"[Task] {problem}\n{summary}"
            new_case = {"query": problem, "task": task, "answer": trajectory}
            self.case_bank.append(new_case)
            embedding = self.encode(task).reshape(1, -1)
            if self.case_embeddings is None:
                self.case_embeddings = embedding
            else:
                self.case_embeddings = torch.cat((self.case_embeddings, embedding), dim=0)
                
        # Bandit Learning
        if len(self.retrieved_cases) != 0:
            metrics = self.bandit.learn(problem, self.retrieved_cases, reward)
            self.metrics.update(metrics)
        
        # Update metric
        metrics = copy.deepcopy(self.metrics)
        self.metrics = {}
        return metrics
    
    @torch.no_grad()
    def encode(self, problem):
        inputs = self.tokenizer([problem], return_tensors='pt').to("cuda")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # L2 normalization
        return embeddings
    
    @torch.no_grad()
    def recall(self, problem):
        q_embedding = self.encode(problem).reshape(1, -1)
        similarity = q_embedding@self.case_embeddings.T
        similarity = similarity.reshape(-1)
        k = min(len(self.case_bank), self.args.recall_size)
        _, index = torch.topk(similarity, k)
        return [self.case_bank[i] for i in index]
        
    def retrieve(self, problem):
        if len(self.case_bank) <= self.args.comk:
            retrieved_cases = [case for case in self.case_bank]
        else:
            # Recall via pretrained embedding model
            recalled_cases = self.recall(problem)
            selected_index, metrics = self.bandit.rank(problem, recalled_cases, self.args.comk)
            retrieved_cases = [recalled_cases[i] for i in selected_index]
            self.metrics.update(metrics)
        return retrieved_cases