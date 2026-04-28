# This file maintains the Python scripts for white-box models in single-turn results (Fig. 4a)

## Qwen3-4B
# 1. Zero-shot
python -u main.py --env ddxplus --agent zero-shot --llm qwen3-4b > zero-shot_ddxplus_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent zero-shot --llm qwen3-4b > zero-shot_mimic-iv-msr_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent zero-shot --llm qwen3-4b > zero-shot_mimic-iv-tlp_qwen3_4b.log 2>&1 &
python -u main.py --env mud --agent zero-shot --llm qwen3-4b > zero-shot_mud_qwen3_4b.log 2>&1 &
python -u main.py --env cmdl --agent zero-shot --llm qwen3-4b > zero-shot_cmdl_qwen3_4b.log 2>&1 &
python -u main.py --env banking77 --agent zero-shot --llm qwen3-4b > zero-shot_banking77_qwen3_4b.log 2>&1 &
python -u main.py --env sentifin --agent zero-shot --llm qwen3-4b > zero-shot_sentifin_qwen3_4b.log 2>&1 &
python -u main.py --env rca --agent zero-shot --llm qwen3-4b > zero-shot_rca_qwen3_4b.log 2>&1 &
python -u main.py --env lfd --agent zero-shot --llm qwen3-4b > zero-shot_lfd_qwen3_4b.log 2>&1 &
python -u main.py --env spider --agent zero-shot --llm qwen3-4b > zero-shot_spider_qwen3_4b.log 2>&1 &
python -u main.py --env bird --agent zero-shot --llm qwen3-4b > zero-shot_bird_qwen3_4b.log 2>&1 &

# 2. NP-CBR
python -u main.py --env ddxplus --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_ddxplus_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_mimic-iv-msr_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_mimic-iv-tlp_qwen3_4b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_mud_qwen3_4b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_cmdl_qwen3_4b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_banking77_qwen3_4b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_sentifin_qwen3_4b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_rca_qwen3_4b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_lfd_qwen3_4b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_spider_qwen3_4b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit Pretrained --llm qwen3-4b > npcbr_bird_qwen3_4b.log 2>&1 &

# 3. REINFORCE+LoRA
python -u main.py --env ddxplus --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_ddxplus_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_mimic-iv-msr_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_mimic-iv-tlp_qwen3_4b.log 2>&1 &
python -u main.py --env mud --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_mud_qwen3_4b.log 2>&1 &
python -u main.py --env cmdl --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_cmdl_qwen3_4b.log 2>&1 &
python -u main.py --env banking77 --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_banking77_qwen3_4b.log 2>&1 &
python -u main.py --env sentifin --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_sentifin_qwen3_4b.log 2>&1 &
python -u main.py --env rca --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_rca_qwen3_4b.log 2>&1 &
python -u main.py --env lfd --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_lfd_qwen3_4b.log 2>&1 &
python -u main.py --env spider --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_spider_qwen3_4b.log 2>&1 &
python -u main.py --env bird --agent reinforce --llm qwen3-4b --learning_rate 1e-4 > reinforce_bird_qwen3_4b.log 2>&1 &

# 4. CASCADE
python -u main.py --env ddxplus --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 0.1 > cascade_ddxplus_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 0.4 > cascade_mimic-iv-msr_qwen3_4b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 5e-6 --nu 0.5 > cascade_mimic-iv-tlp_qwen3_4b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 1e-5 --nu 0.1 > cascade_mud_qwen3_4b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 1e-5 --nu 0.2 > cascade_cmdl_qwen3_4b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 0.2 > cascade_banking77_qwen3_4b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 5e-6 --nu 0.6 > cascade_sentifin_qwen3_4b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 0.6 > cascade_rca_qwen3_4b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 1e-5 --nu 0.3 > cascade_lfd_qwen3_4b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 0.4 > cascade_spider_qwen3_4b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit NeuralLinLogUCB --llm qwen3-4b --learning_rate 2e-5 --nu 1.0 > cascade_bird_qwen3_4b.log 2>&1 &

# Qwen3-8B
# 1. Zero-shot
python -u main.py --env ddxplus --agent zero-shot --llm qwen3-8b > zero-shot_ddxplus_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent zero-shot --llm qwen3-8b > zero-shot_mimic-iv-mr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent zero-shot --llm qwen3-8b > zero-shot_mimic-iv-msr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent zero-shot --llm qwen3-8b > zero-shot_mimic-iv-tlp_qwen3_8b.log 2>&1 &
python -u main.py --env mud --agent zero-shot --llm qwen3-8b > zero-shot_mud_qwen3_8b.log 2>&1 &
python -u main.py --env cmdl --agent zero-shot --llm qwen3-8b > zero-shot_cmdl_qwen3_8b.log 2>&1 &
python -u main.py --env banking77 --agent zero-shot --llm qwen3-8b > zero-shot_banking77_qwen3_8b.log 2>&1 &
python -u main.py --env sentifin --agent zero-shot --llm qwen3-8b > zero-shot_sentifin_qwen3_8b.log 2>&1 &
python -u main.py --env rca --agent zero-shot --llm qwen3-8b > zero-shot_rca_qwen3_8b.log 2>&1 &
python -u main.py --env lfd --agent zero-shot --llm qwen3-8b > zero-shot_lfd_qwen3_8b.log 2>&1 &
python -u main.py --env spider --agent zero-shot --llm qwen3-8b > zero-shot_spider_qwen3_8b.log 2>&1 &
python -u main.py --env bird --agent zero-shot --llm qwen3-8b > zero-shot_bird_qwen3_8b.log 2>&1 &

# 2. NP-CBR
python -u main.py --env ddxplus --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_ddxplus_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_mimic-iv-mr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_mimic-iv-msr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_mimic-iv-tlp_qwen3_8b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_mud_qwen3_8b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_cmdl_qwen3_8b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_banking77_qwen3_8b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_sentifin_qwen3_8b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_rca_qwen3_8b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_lfd_qwen3_8b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_spider_qwen3_8b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit Pretrained --llm qwen3-8b > npcbr_bird_qwen3_8b.log 2>&1 &

# 3. REINFORCE+LoRA
python -u main.py --env ddxplus --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_ddxplus_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent reinforce --llm qwen3-8b --learning_rate 1e-5 > reinforce_mimic-iv-mr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_mimic-iv-msr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_mimic-iv-tlp_qwen3_8b.log 2>&1 &
python -u main.py --env mud --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_mud_qwen3_8b.log 2>&1 &
python -u main.py --env cmdl --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_cmdl_qwen3_8b.log 2>&1 &
python -u main.py --env banking77 --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_banking77_qwen3_8b.log 2>&1 &
python -u main.py --env sentifin --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_sentifin_qwen3_8b.log 2>&1 &
python -u main.py --env rca --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_rca_qwen3_8b.log 2>&1 &
python -u main.py --env lfd --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_lfd_qwen3_8b.log 2>&1 &
python -u main.py --env spider --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_spider_qwen3_8b.log 2>&1 &
python -u main.py --env bird --agent reinforce --llm qwen3-8b --learning_rate 1e-4 > reinforce_bird_qwen3_8b.log 2>&1 &

# 4. CASCADE
python -u main.py --env ddxplus --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.2 > cascade_ddxplus_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.1 > cascade_mimic-iv-mr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.5 > cascade_mimic-iv-msr_qwen3_8b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 5e-6 --nu 0.1 > cascade_mimic-iv-tlp_qwen3_8b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 1e-6 --nu 0.9 > cascade_mud_qwen3_8b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.8 > cascade_cmdl_qwen3_8b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.1 > cascade_banking77_qwen3_8b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.1 > cascade_sentifin_qwen3_8b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 1e-5 --nu 1.0 > cascade_rca_qwen3_8b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 0.5 > cascade_lfd_qwen3_8b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 2e-5 --nu 1.0 > cascade_spider_qwen3_8b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit NeuralLinLogUCB --llm qwen3-8b --learning_rate 1e-6 --nu 0.5 > cascade_bird_qwen3_8b.log 2>&1 &

# Qwen3-14B
# 1. Zero-shot
python -u main.py --env ddxplus --agent zero-shot --llm qwen3-14b > zero-shot_ddxplus_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent zero-shot --llm qwen3-14b > zero-shot_mimic-iv-mr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent zero-shot --llm qwen3-14b > zero-shot_mimic-iv-msr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent zero-shot --llm qwen3-14b > zero-shot_mimic-iv-tlp_qwen3_14b.log 2>&1 &
python -u main.py --env mud --agent zero-shot --llm qwen3-14b > zero-shot_mud_qwen3_14b.log 2>&1 &
python -u main.py --env cmdl --agent zero-shot --llm qwen3-14b > zero-shot_cmdl_qwen3_14b.log 2>&1 &
python -u main.py --env banking77 --agent zero-shot --llm qwen3-14b > zero-shot_banking77_qwen3_14b.log 2>&1 &
python -u main.py --env sentifin --agent zero-shot --llm qwen3-14b > zero-shot_sentifin_qwen3_14b.log 2>&1 &
python -u main.py --env rca --agent zero-shot --llm qwen3-14b > zero-shot_rca_qwen3_14b.log 2>&1 &
python -u main.py --env lfd --agent zero-shot --llm qwen3-14b > zero-shot_lfd_qwen3_14b.log 2>&1 &
python -u main.py --env spider --agent zero-shot --llm qwen3-14b > zero-shot_spider_qwen3_14b.log 2>&1 &
python -u main.py --env bird --agent zero-shot --llm qwen3-14b > zero-shot_bird_qwen3_14b.log 2>&1 &

# 2. NP-CBR
python -u main.py --env ddxplus --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_ddxplus_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_mimic-iv-mr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_mimic-iv-msr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_mimic-iv-tlp_qwen3_14b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_mud_qwen3_14b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_cmdl_qwen3_14b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_banking77_qwen3_14b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_sentifin_qwen3_14b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_rca_qwen3_14b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_lfd_qwen3_14b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_spider_qwen3_14b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit Pretrained --llm qwen3-14b > npcbr_bird_qwen3_14b.log 2>&1 &

# 3. REINFORCE+LoRA
python -u main.py --env ddxplus --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_ddxplus_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent reinforce --llm qwen3-14b --learning_rate 1e-5 > reinforce_mimic-iv-mr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_mimic-iv-msr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_mimic-iv-tlp_qwen3_14b.log 2>&1 &
python -u main.py --env mud --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_mud_qwen3_14b.log 2>&1 &
python -u main.py --env cmdl --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_cmdl_qwen3_14b.log 2>&1 &
python -u main.py --env banking77 --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_banking77_qwen3_14b.log 2>&1 &
python -u main.py --env sentifin --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_sentifin_qwen3_14b.log 2>&1 &
python -u main.py --env rca --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_rca_qwen3_14b.log 2>&1 &
python -u main.py --env lfd --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_lfd_qwen3_14b.log 2>&1 &
python -u main.py --env spider --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_spider_qwen3_14b.log 2>&1 &
python -u main.py --env bird --agent reinforce --llm qwen3-14b --learning_rate 1e-4 > reinforce_bird_qwen3_14b.log 2>&1 &

# 4. CASCADE
python -u main.py --env ddxplus --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 2e-5 --nu 0.1 > cascade_ddxplus_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-mr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 2e-5 --nu 0.3 > cascade_mimic-iv-mr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-msr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 1e-5 --nu 0.4 > cascade_mimic-iv-msr_qwen3_14b.log 2>&1 &
python -u main.py --env mimic-iv-tlp --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 5e-6 --nu 0.1 > cascade_mimic-iv-tlp_qwen3_14b.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 1e-6 --nu 0.4 > cascade_mud_qwen3_14b.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 1e-5 --nu 0.7 > cascade_cmdl_qwen3_14b.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 2e-5 --nu 0.1 > cascade_banking77_qwen3_14b.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 2e-5 --nu 0.4 > cascade_sentifin_qwen3_14b.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 2e-5 --nu 0.3 > cascade_rca_qwen3_14b.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 5e-6 --nu 0.1 > cascade_lfd_qwen3_14b.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 5e-6 --nu 0.3 > cascade_spider_qwen3_14b.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit NeuralLinLogUCB --llm qwen3-14b --learning_rate 1e-6 --nu 0.9 > cascade_bird_qwen3_14b.log 2>&1 &


