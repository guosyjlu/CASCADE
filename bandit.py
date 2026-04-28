import torch
import numpy as np
import torch.nn as nn
from utils import inv_sherman_morrison
from modeling import ModernBert
from transformers import ModernBertForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from torch.func import vmap, grad, functional_call

class PretrainedBandit:
    """
    This is the baseline: pretrained reranker model as the bandit policy.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path)
        self.model = ModernBertForSequenceClassification.from_pretrained(args.reranker_model_path).to("cuda")
        self.model.eval()
    
    @torch.no_grad()
    def rank(self, problem, retrieved_cases, topk):
        context = [[problem, case["task"]] for case in retrieved_cases]
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
        outputs = self.model(**inputs, return_dict=True)
        sample_r = outputs.logits.reshape(-1).float().cpu().numpy()
        arm = np.argsort(-sample_r)[:topk]
        return arm, {}
    
    def learn(self, problem, retrieved_cases, reward):
        return {}

    
class NeuralLinLogUCB:
    """
    The proposed Neural-LinLogUCB algorithm.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path)
        self.model = ModernBert.from_pretrained(args.reranker_model_path).to("cuda")
        
        # Zero and freeze the bias of the linear classifier head
        nn.init.zeros_(self.model.classifier.bias)
        self.model.classifier.bias.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.input_dim = 768

        # Buffer for neural model training
        self.context_buffer = []
        self.reward_buffer = []
        
        # Buffer for linear model training
        self.all_context = []
        self.context_features = []
        self.prediction_target = []
        
        # Setting
        self.lamdba = args.lamdba
        self.nu = args.nu
        self.A_inv = np.eye(self.input_dim) / self.lamdba
        self.theta = self.model.classifier.weight.detach().cpu().numpy()
    
    @torch.no_grad()
    def rank(self, problem, retrieved_cases, topk):
        self.model.eval()
        context = [[problem, case["task"]] for case in retrieved_cases]
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
        outputs = self.model(**inputs, return_dict=True)
        context_feature = outputs.features.reshape(-1, self.input_dim).float().cpu().numpy()
        n_arm = context_feature.shape[0]
        exploitation_term = expit(context_feature.reshape(n_arm, -1) @ self.theta.reshape(-1))
        exploration_term = np.sqrt(
            np.einsum("ij,jk,ik->i", context_feature, self.A_inv, context_feature)
        )
        sample_r = exploitation_term + self.nu * exploration_term
        arm = np.argsort(-sample_r)[:topk]
        
        # Record logs
        metrics = dict()
        metrics["UCB/ucb_max"] = sample_r.max()
        metrics["UCB/ucb_min"] = sample_r.min()
        metrics["UCB/ucb_mean"] = sample_r.mean()
        metrics["UCB/explore_max"] = exploration_term.max()
        metrics["UCB/explore_min"] = exploration_term.min()
        metrics["UCB/explore_mean"] = exploration_term.mean()
        metrics["UCB/explore_chosen_max"] = exploration_term[arm].max()
        metrics["UCB/exploit_max"] = exploitation_term.max()
        metrics["UCB/exploit_min"] = exploitation_term.min()
        metrics["UCB/exploit_mean"] = exploitation_term.mean()
        metrics["UCB/exploit_chosen_max"] = exploitation_term[arm].max()
        return arm, metrics
    
    def fit_theta(self, X, y):        
        p = expit(X.reshape(-1, self.input_dim) @ self.theta.reshape(self.input_dim))
        grad = X.T @ (p - y) / X.shape[0]
        self.theta = self.theta - self.args.learning_rate / self.args.batch_size * grad
        return {}
        
    def update_A_inv(self, context_feature):
        n_arm = context_feature.shape[0]
        for i in range(n_arm):
            self.A_inv = inv_sherman_morrison(context_feature[i].reshape(-1), self.A_inv)
        
    def reset_A_inv(self):
        context_features = np.array(self.context_features)
        A = context_features.T @ context_features + np.eye(self.input_dim) * self.lamdba
        self.A_inv = np.linalg.inv(A)
    
    def update_neural_model(self):
        torch.cuda.empty_cache()
        self.model.train()
        indices = torch.randperm(len(self.context_buffer))[: self.args.batch_size * self.args.comk]
        self.optimizer.zero_grad()
        for i in range(self.args.comk):
            batch_indices = indices[i * self.args.batch_size : (i + 1) * self.args.batch_size]
            batch_context = [self.context_buffer[j] for j in batch_indices]
            batch_reward = torch.tensor([self.reward_buffer[j] for j in batch_indices]).reshape(-1, 1).float().to("cuda")
                
            inputs = self.tokenizer(batch_context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits.reshape(-1, 1)
            
            loss = nn.BCEWithLogitsLoss()(logits, batch_reward) / self.args.comk
            print(f"[{i+1}/{self.args.comk}] Loss = {loss.item():.6f}")
            loss.backward()
            if (i + 1) == self.args.comk:
                self.optimizer.step()
        
    @torch.no_grad()    
    def reset_context_feature(self):
        context_feature_list = []
        self.model.eval()
        for i in range(0, len(self.all_context), 512):
            context = self.all_context[i: i+512]
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim).float().cpu().numpy()
            context_feature_list.append(context_feature)
        context_features = np.concatenate(context_feature_list, axis=0)
        self.context_features = context_features.tolist()
      
    def learn(self, problem, retrieved_cases, reward):
        context = [[problem, case["task"]] for case in retrieved_cases]
        self.context_buffer += context
        self.all_context += context
        rewards = [reward] * len(retrieved_cases)
        self.reward_buffer += rewards
        self.prediction_target += rewards
        
        with torch.no_grad():
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim).float().cpu().numpy()
        self.context_features.append(context_feature)
        
        if len(self.context_buffer) >= self.args.batch_size * self.args.comk:
            self.update_neural_model()
            self.reset_context_feature()
            # Clear the training buffer
            self.context_buffer = []
            self.reward_buffer = []
            # Fit theta and reset A_inv
            self.theta = self.model.classifier.weight.detach().cpu().numpy()
            self.reset_A_inv()
        else:
            # Fit theta and update A_inv
            self.fit_theta(context_feature, reward)
            self.update_A_inv(context_feature)
        return {}


class LinearLogisticUCB:
    """
    Linear Logistic UCB: LinLogUCB
    Reference: LinearGLM from paper "Neural Dueling Bandits: Preference-Based Optimization with Human Feedback", ICLR 2025.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path)
        self.model = ModernBert.from_pretrained(args.reranker_model_path).to("cuda")
        self.model.eval()
        
        self.input_dim = 768
        self.lamdba = args.lamdba
        self.nu = args.nu
        
        # Initial variables for storing information
        self.samples    = 0                         # Number of samples
        self.X          = []                        # Context-actions feature vectors
        self.y          = []
       
        # Initialialization of gram matrix if features are fixed
        self.V      = self.lamdba * np.identity(self.input_dim)
        
        # Initializing model parameters
        self.theta  = np.ones(self.input_dim) / self.input_dim
    
    @torch.no_grad()
    def rank(self, problem, retrieved_cases, topk):
        self.model.eval()
        context = [[problem, case["task"]] for case in retrieved_cases]
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
        outputs = self.model(**inputs, return_dict=True)
        context_feature = outputs.features.reshape(-1, self.input_dim).float().cpu().numpy()
        n_arm = context_feature.shape[0]
        exploitation_term = expit(context_feature.reshape(n_arm, -1) @ self.theta.reshape(-1))
        exploration_term = np.sqrt(
            np.einsum("ij,jk,ik->i", context_feature, np.linalg.inv(self.V), context_feature)
        )
        sample_r = exploitation_term + self.nu * exploration_term
        arm = np.argsort(-sample_r)[:topk]
        return arm, {}
      
    def learn(self, problem, retrieved_cases, reward):
        n_arm = len(retrieved_cases)
        context = [[problem, case["task"]] for case in retrieved_cases]
        
        with torch.no_grad():
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim).float().cpu().numpy()

        for i in range(n_arm):
            self.X.append(context_feature[i])
            self.y.append(reward)
            self.V += np.outer(context_feature[i], context_feature[i])
        
        # Update theta
        X = np.array(self.X)
        y = np.array(self.y)
        if len(np.unique(y)) == 2:
            clf = LogisticRegression(
                fit_intercept=False,
                penalty=None,
                solver='lbfgs',
                tol=1e-4,
                max_iter=1000
            )
            clf.fit(X, y)
            theta = clf.coef_.flatten()    
            self.theta = theta
        return {}

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class NeuralLogUCB:
    """
    Neural Logistic UCB: NeuralLogUCB
    Reference: NeuralGLM from paper "Neural Dueling Bandits: Preference-Based Optimization with Human Feedback", ICLR 2025.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path)
        self.model = ModernBert.from_pretrained(args.reranker_model_path).to("cuda")
        self.model.eval()
        
        self.input_dim = 768
        self.lamdba = args.lamdba
        self.nu = args.nu
        
        # Initial variables for storing information
        self.context_list = []
        self.reward = []
        
        # Initializing neural network model with pytorch
        self.func = Network(self.input_dim, hidden_size=100).cuda()
        
        # Storing the initial state of the NN
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = args.lamdba * torch.ones((self.total_param,)).cuda()
    
    def rank(self, problem, retrieved_cases, topk):
        self.model.eval()
        context = [[problem, case["task"]] for case in retrieved_cases]
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
        outputs = self.model(**inputs, return_dict=True)
        context_feature = outputs.features.reshape(-1, self.input_dim)
     
        param_names, param_tensors = zip(*self.func.named_parameters())
        params = {n: p for n, p in zip(param_names, param_tensors)}

        def model_scalar_out(p, x):
            y = functional_call(self.func, p, (x.unsqueeze(0) if x.dim()==1 else x,))
            return y.squeeze(-1).squeeze(0)

        def grad_vector_for_one(x):
            gdict = grad(model_scalar_out)(params, x)
            return torch.cat([gdict[n].reshape(-1) for n in param_names], dim=0)

        G = vmap(grad_vector_for_one)(context_feature)
        g_list = [g.detach().cpu() for g in G]

        with torch.no_grad():
            mu = self.func(context_feature).squeeze(-1)

        U = self.U
        if U.ndim == 2 and U.shape[0] == 1:
            U = U.squeeze(0)
        if U.ndim == 0:
            sigma2 = self.lamdba * self.nu * (G**2) / U
        else:
            sigma2 = self.lamdba * self.nu * (G**2) / U

        sigma = torch.sqrt(sigma2.sum(dim=1))
        sampled = (mu.detach() + sigma.detach()).tolist()
            
        sample_r = np.array(sampled)
        arm = np.argsort(-sample_r)[:topk]
        for a in arm:
            self.U += g_list[a].cuda() * g_list[a].cuda()
        return arm, {}
      
    def learn(self, problem, retrieved_cases, reward):
        n_arm = len(retrieved_cases)
        context = [[problem, case["task"]] for case in retrieved_cases]
        
        with torch.no_grad():
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim)
        for i in range(n_arm):
            self.context_list.append(context_feature[i].reshape(1, -1))
            self.reward.append(reward)
        
        context_tensor = torch.cat(self.context_list, dim=0).to("cuda")
        reward_tensor = torch.tensor(self.reward, dtype=torch.float32).to("cuda")
        
        optimizer = torch.optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)        
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = context_tensor[idx]
                r = reward_tensor[idx].unsqueeze(0)
                optimizer.zero_grad()
                logits = self.func(c.cuda())
                loss = nn.BCEWithLogitsLoss()(logits, r)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return {}
            if batch_loss / length <= 1e-3:
                return {}



class NeuralLinUCB:
    """
    Neural-Linear UCB: NeuralLinUCB
    Reference: Neural Contextual Bandits with Deep Representation and Shallow Exploration, ICLR 2022.
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_path)
        self.model = ModernBert.from_pretrained(args.reranker_model_path).to("cuda")
        
        # Zero and freeze the bias of the linear classifier head
        nn.init.zeros_(self.model.classifier.bias)
        self.model.classifier.bias.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.input_dim = 768

        # Buffer for neural model training
        self.context_buffer = []
        self.reward_buffer = []
        
        # Buffer for linear model training
        self.all_context = []
        self.all_reward = []
        self.context_features = []
        self.prediction_target = []
        
        # Setting
        self.lamdba = args.lamdba
        self.nu = args.nu
        
        # Linear head
        self.theta = np.random.uniform(-1, 1, self.input_dim)
        self.b = np.zeros(self.input_dim)
        self.A_inv = np.eye(self.input_dim) / args.lamdba
    
    @torch.no_grad()
    def rank(self, problem, retrieved_cases, topk):
        self.model.eval()
        context = [[problem, case["task"]] for case in retrieved_cases]
        inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
        outputs = self.model(**inputs, return_dict=True)
        context_feature = outputs.features.reshape(-1, self.input_dim).float().cpu().numpy()
        n_arm = context_feature.shape[0]
        exploitation_term = np.array([np.dot(context_feature[a, :], self.theta) for a in range(n_arm)])
        exploration_term = np.array([np.sqrt(np.dot(context_feature[a, :], np.dot(self.A_inv, context_feature[a, :].T))) for a in range(n_arm)])
        sample_r = exploitation_term + self.nu * exploration_term
        arm = np.argsort(-sample_r)[:topk]
        
        # Record logs
        metrics = dict()
        metrics["UCB/ucb_max"] = sample_r.max()
        metrics["UCB/ucb_min"] = sample_r.min()
        metrics["UCB/ucb_mean"] = sample_r.mean()
        metrics["UCB/explore_max"] = exploration_term.max()
        metrics["UCB/explore_min"] = exploration_term.min()
        metrics["UCB/explore_mean"] = exploration_term.mean()
        metrics["UCB/explore_chosen_max"] = exploration_term[arm].max()
        metrics["UCB/exploit_max"] = exploitation_term.max()
        metrics["UCB/exploit_min"] = exploitation_term.min()
        metrics["UCB/exploit_mean"] = exploitation_term.mean()
        metrics["UCB/exploit_chosen_max"] = exploitation_term[arm].max()
        return arm, metrics
    
    def update_linear_model(self, X, y):
        arm_num = X.shape[0]       
        for a in range(arm_num):
            self.b += X[a] * y
            self.A_inv = inv_sherman_morrison(X[a, :], self.A_inv)
        self.theta = np.matmul(self.A_inv, self.b)
        return {}
        
    def reset_linear_model(self):
        context_features = np.array(self.context_features)
        rewards = np.array(self.all_reward)
        A = context_features.T @ context_features + np.eye(self.input_dim) * self.lamdba
        self.A_inv = np.linalg.inv(A)
        self.b = (context_features.T @ rewards).reshape(-1)
        self.theta = np.matmul(self.A_inv, self.b)
    
    def update_neural_model(self):
        torch.cuda.empty_cache()
        self.model.train()
        indices = torch.randperm(len(self.context_buffer))[: self.args.batch_size * self.args.comk]
        self.optimizer.zero_grad()
        for i in range(self.args.comk):
            batch_indices = indices[i * self.args.batch_size : (i + 1) * self.args.batch_size]
            batch_context = [self.context_buffer[j] for j in batch_indices]
            batch_reward = torch.tensor([self.reward_buffer[j] for j in batch_indices]).reshape(-1, 1).float().to("cuda")
                
            inputs = self.tokenizer(batch_context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits.reshape(-1, 1)
            
            loss = nn.MSELoss()(logits, batch_reward) / self.args.comk
            print(f"[{i+1}/{self.args.comk}] Loss = {loss.item():.6f}")
            loss.backward()
            if (i + 1) == self.args.comk:
                self.optimizer.step()
        
    @torch.no_grad()    
    def reset_context_feature(self):
        context_feature_list = []
        self.model.eval()
        for i in range(0, len(self.all_context), 512):
            context = self.all_context[i: i+512]
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim).float().cpu().numpy()
            context_feature_list.append(context_feature)
        context_features = np.concatenate(context_feature_list, axis=0)
        self.context_features = context_features.tolist()
      
    def learn(self, problem, retrieved_cases, reward):
        context = [[problem, case["task"]] for case in retrieved_cases]
        self.context_buffer += context
        self.all_context += context
        rewards = [reward] * len(retrieved_cases)
        self.reward_buffer += rewards
        self.all_reward += rewards
        self.prediction_target += rewards
        
        with torch.no_grad():
            inputs = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=2048).to("cuda")
            outputs = self.model(**inputs, return_dict=True)
            context_feature = outputs.features
            context_feature = context_feature.reshape(-1, self.input_dim).float().cpu().numpy()
        self.context_features.append(context_feature)
        
        if len(self.context_buffer) >= self.args.batch_size * self.args.comk:
            self.update_neural_model()
            self.reset_context_feature()
            # Clear the training buffer
            self.context_buffer = []
            self.reward_buffer = []
            # Fit theta and reset A_inv
            self.reset_linear_model()
        else:
            # Fit theta and update A_inv
            self.update_linear_model(context_feature, reward)
        return {}