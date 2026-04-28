import wandb
import asyncio
from llm import *
from config import get_config
from agent import DeepResearchCBRAgent, DeepResearchZeroShotAgent
from env import TwoWikiEnv


async def run_one_seed(args):
    try:
        wandb_config = {k: str(v) if isinstance(v, float) else v for k, v in vars(args).items()}
        run_name = args.bandit if args.agent == "cbr" else args.agent
        project_name = args.env if args.comk == 1 else f"{args.env}-k"
        project_name = project_name+"-web" if args.web_search else project_name
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            settings=wandb.Settings(console="off", ignore_globs=["*.*"])
        )
            
        # Load environemnt
        env = TwoWikiEnv()
        await env.connect_to_servers(mode="online" if args.web_search else "local")
        # Load agent
        if args.agent == "zero-shot":
            agent = DeepResearchZeroShotAgent(args, env)
        elif args.agent == "cbr":
            agent = DeepResearchCBRAgent(args, env)
        else:
            raise NotImplementedError("The specified agent type is not supported yet.")
            
        # Online learning
        acc_reward, acc_regret = 0.0, 0.0
        done = env.reset()
        while not done:
            # 1. Observe the query
            query = env.observe()
            # 2. Derive the plan
            msgs = agent.solve(query)
            # 3. Interact with the env
            result_infos = await env.interact(msgs, args)
            final_answer = result_infos["answer"]
            # 4. Receive the reward
            reward, judgement = env.reward_function(final_answer)
            # 5. Update the agent
            metrics = agent.learn(query, result_infos["summary"], result_infos["trajectory"], reward)
            # 6. Update statistics and metrics
            acc_reward += reward
            acc_regret += (1.0 - reward)
            metrics["Success Rate"] = acc_reward / (env.index + 1)
            metrics["Regret"] = acc_regret
            metrics["Reward"] = acc_reward
            # 7. Logging
            wandb.log(metrics, step=env.index+1)
            # 8. Move to the next step
            done = env.step()
            # 9. Log detailed info
            infos = {
                "query": query,
                "final_answer": final_answer,
                "reward": reward,
                "judgement": judgement,
                "summary": result_infos["summary"],
                "trajectory": result_infos["trajectory"],
            }
        run.finish()
    finally:
        await env.cleanup()

async def main():
    args = get_config()
    seed_list = [0, 1, 2, 3, 4] if args.seed == -1 else [args.seed]  
    for seed in seed_list:
        args.seed = seed
        await run_one_seed(args)

if __name__ == '__main__':
    asyncio.run(main())