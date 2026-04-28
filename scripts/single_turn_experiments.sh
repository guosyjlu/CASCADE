# This file maintains the Python scripts for the main experiments in single-turn results (Fig. 3)

# 1. Zero-shot
python -u main.py --env ddxplus --agent zero-shot --llm qwen3-32b > zero-shot_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent zero-shot --llm qwen3-32b > zero-shot_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent zero-shot --llm qwen3-32b > zero-shot_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent zero-shot --llm qwen3-32b > zero-shot_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent zero-shot --llm qwen3-32b > zero-shot_mud.log 2>&1 &
python -u main.py --env cmdl --agent zero-shot --llm qwen3-32b > zero-shot_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent zero-shot --llm qwen3-32b > zero-shot_banking77.log 2>&1 &
python -u main.py --env sentifin --agent zero-shot --llm qwen3-32b > zero-shot_sentifin.log 2>&1 &
python -u main.py --env rca --agent zero-shot --llm qwen3-32b > zero-shot_rca.log 2>&1 &
python -u main.py --env lfd --agent zero-shot --llm qwen3-32b > zero-shot_lfd.log 2>&1 &
python -u main.py --env spider --agent zero-shot --llm qwen3-32b > zero-shot_spider.log 2>&1 &
python -u main.py --env bird --agent zero-shot --llm qwen3-32b > zero-shot_bird.log 2>&1 &

# 2. ICRL
python -u main.py --env ddxplus --agent icrl --llm qwen3-32b > icrl_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent icrl --llm qwen3-32b > icrl_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent icrl --llm qwen3-32b > icrl_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent icrl --llm qwen3-32b > icrl_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent icrl --llm qwen3-32b > icrl_mud.log 2>&1 &
python -u main.py --env cmdl --agent icrl --llm qwen3-32b > icrl_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent icrl --llm qwen3-32b > icrl_banking77.log 2>&1 &
python -u main.py --env sentifin --agent icrl --llm qwen3-32b > icrl_sentifin.log 2>&1 &
python -u main.py --env rca --agent icrl --llm qwen3-32b > icrl_rca.log 2>&1 &
python -u main.py --env lfd --agent icrl --llm qwen3-32b > icrl_lfd.log 2>&1 &
python -u main.py --env spider --agent icrl --llm qwen3-32b > icrl_spider.log 2>&1 &
python -u main.py --env bird --agent icrl --llm qwen3-32b > icrl_bird.log 2>&1 &

# 3. ICRLPlus
python -u main.py --env ddxplus --agent icrl-plus --llm qwen3-32b > icrl-plus_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent icrl-plus --llm qwen3-32b > icrl-plus_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent icrl-plus --llm qwen3-32b > icrl-plus_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent icrl-plus --llm qwen3-32b > icrl-plus_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent icrl-plus --llm qwen3-32b > icrl-plus_mud.log 2>&1 &
python -u main.py --env cmdl --agent icrl-plus --llm qwen3-32b > icrl-plus_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent icrl-plus --llm qwen3-32b > icrl-plus_banking77.log 2>&1 &
python -u main.py --env sentifin --agent icrl-plus --llm qwen3-32b > icrl-plus_sentifin.log 2>&1 &
python -u main.py --env rca --agent icrl-plus --llm qwen3-32b > icrl-plus_rca.log 2>&1 &
python -u main.py --env lfd --agent icrl-plus --llm qwen3-32b > icrl-plus_lfd.log 2>&1 &
python -u main.py --env spider --agent icrl-plus --llm qwen3-32b > icrl-plus_spider.log 2>&1 &
python -u main.py --env bird --agent icrl-plus --llm qwen3-32b > icrl-plus_bird.log 2>&1 &

# 4. NP-CBR
python -u main.py --env ddxplus --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_mud.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_banking77.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_sentifin.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_rca.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_lfd.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_spider.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_bird.log 2>&1 &

# 5. REINFORCE+LoRA
python -u main.py --env ddxplus --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent reinforce --llm qwen3-32b --learning_rate 1e-5 > reinforce_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_mud.log 2>&1 &
python -u main.py --env cmdl --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_banking77.log 2>&1 &
python -u main.py --env sentifin --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_sentifin.log 2>&1 &
python -u main.py --env rca --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_rca.log 2>&1 &
python -u main.py --env lfd --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_lfd.log 2>&1 &
python -u main.py --env spider --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_spider.log 2>&1 &
python -u main.py --env bird --agent reinforce --llm qwen3-32b --learning_rate 1e-4 > reinforce_bird.log 2>&1 &

# 6. CASCADE
python -u main.py --env ddxplus --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.1 > cascade_ddxplus.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.5 > cascade_mimic-iv-mr.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.4 > cascade_mimic-iv-msr.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.1 > cascade_mimic-iv-tlp.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.8 > cascade_mud.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.2 > cascade_cmdl.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.3 > cascade_banking77.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.1 > cascade_sentifin.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.3 > cascade_rca.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.4 > cascade_lfd.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 5e-6 --nu 0.8 > cascade_spider.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 5e-6 --nu 0.3 > cascade_bird.log 2>&1 &

