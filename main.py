import wandb
from config import get_config
from llm import *
from agent import ZeroShotAgent, CBRAgent, ICRLAgent, ICRLPlusAgent, REINFORCEAgent, CBRGTAgent
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

    # Load wandb
    wandb_config = {k: str(v) if isinstance(v, float) else v for k, v in vars(args).items()}
    run_name = args.bandit if args.agent == "cbr" else args.agent
    project_name = args.env if args.comk == 1 else f"{args.env}-k"
    if args.agent == "cbrgt":
        run_name = args.bandit
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
    if args.agent == "zero-shot":
        agent = ZeroShotAgent(args, env)
    elif args.agent == "cbr":
        agent = CBRAgent(args, env)
    elif args.agent == "cbrgt":
        agent = CBRGTAgent(args, env)
    elif args.agent == "icrl":
        agent = ICRLAgent(args, env)
    elif args.agent == "icrl-plus":
        agent = ICRLPlusAgent(args, env)
    elif args.agent == "reinforce":
        agent = REINFORCEAgent(args, env)
    else:
        raise NotImplementedError("The specified agent type is not supported yet.")
        
    # Deployment-time learning
    acc_reward, acc_regret = 0.0, 0.0
    done = env.reset()
    while not done:
        # 1. Observe the query
        query = env.observe()
        # 2. Solve the query to derive the prompt 
        prompt = agent.solve(query)
        # 3. Sample and evaluate the answer
        if env.type == "single-turn":
            if args.agent == "reinforce":
                generated_text = agent.generate_response(prompt)
            else:
                generated_text = query_LLM(args, prompt)
            generated_answer, reward = env.evaluate(generated_text)
        elif env.type == "multi-turn":
            obs, reward, stop, action_list = env.reset_env()
            while not stop:
                suffix = env.get_prompt(obs, action_list)
                temp_prompt = prompt + suffix
                generated_text = query_LLM(args, temp_prompt)
                obs, reward, stop, action_list = env.interact(generated_text)
            generated_answer = env.get_trajectory()
        else:
            raise NotImplementedError("Unknown type of environment.")
        # 4. Update the agent
        metrics = agent.learn(query, generated_answer, reward)
        # 5. Update statistics and metrics
        acc_reward += reward
        acc_regret += (1.0 - reward)
        metrics["Success Rate"] = acc_reward / (env.index + 1)
        metrics["Regret"] = acc_regret
        metrics["Reward"] = acc_reward
        # 6. Logging
        wandb.log(metrics, step=env.index+1)
        # 7. Move to the next step
        done = env.step()
        
    # Release wandb and collect CUDA memory
    if args.agent == "reinforce":
        agent.cleanup()
    run.finish()

if __name__ == '__main__':
    args = get_config()
    seed_list = [0, 1, 2, 3, 4] if args.seed == -1 else [args.seed]  
    for seed in seed_list:
        run_one_seed(args, seed)