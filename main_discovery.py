import wandb
from llm import *
from config import get_config
from agent import CBRDiscoveryAgent
from env import ENV_DICT

def query_LLM(args, prompt):
    if args.serving_mode == "openai":
        return query_openai(prompt, model=args.llm)
    elif args.serving_mode == "vllm":
        return query_vllm_openai(prompt, model=args.llm, server=args.server, port=args.port, max_completion_tokens=args.max_new_tokens)
    else:
        raise NotImplementedError("The specified LLM is not supported yet.")

def run_one_seed(args, seed):
    args.seed = seed
    wandb_config = {k: str(v) if isinstance(v, float) else v for k, v in vars(args).items()}
    assert args.agent == "cbrdiscover"
    run_name = args.bandit + args.discover_strategy
    project_name = f"{args.env}-{args.agent}"
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config,
        settings=wandb.Settings(console="off", ignore_globs=["*.*"])
    )
        
    # Load environemnt
    env = ENV_DICT[args.env]()
    # Load agent
    agent = CBRDiscoveryAgent(args, env)
        
    # Deployment-Time learning
    acc_reward, acc_regret = 0.0, 0.0
    acc_discover, num_discover = 0.0, 0.0
    done = env.reset()
    while not done:
        # 1. Observe the query
        query = env.observe()
        # 2. Solve the query 
        isDicovery, prompt = agent.solve(query)
        # 3. Evaluate the answer
        assert env.type == "single-turn"
        generated_text = query_LLM(args, prompt)
        generated_answer, reward = env.evaluate(generated_text)
        # 4. Update the agent
        metrics = agent.learn(query, generated_answer, reward, isDicovery)
        # 5. Update statistics and metrics
        logged_reward = 1 if isDicovery else reward
        acc_reward += logged_reward
        acc_regret += (1.0 - logged_reward)
        acc_discover += 1.0 if isDicovery and reward == 0.0 else 0.0
        num_discover += 1.0 if isDicovery else 0.0
        metrics["Success Rate"] = acc_reward / (env.index + 1)
        metrics["Regret"] = acc_regret
        metrics["Reward"] = acc_reward
        metrics["NumDiscovery"] = num_discover
        if num_discover > 0:
            metrics["DiscoverSuccessRate"] = acc_discover / num_discover
        # 6. Logging
        wandb.log(metrics, step=env.index+1)
        # 7. Move to the next step
        done = env.step()

    run.finish()

if __name__ == '__main__':
    args = get_config()
    seed_list = [0, 1, 2, 3, 4] if args.seed == -1 else [args.seed]  
    for seed in seed_list:
        run_one_seed(args, seed)
        
        