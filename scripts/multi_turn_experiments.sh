# This file maintains the Python scripts for the main experiments in multi-turn, simulated results (Fig. 5)

# ALFWorld
# Please configure the <YOUR_ALFWORLD_PATH> as the environment path in env/alfworld.py
python -u main.py --env alfworld --agent zero-shot --llm qwen3-32b > zero-shot_alfworld.log 2>&1 &
python -u main.py --env alfworld --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_alfworld.log 2>&1 &
python -u main.py --env alfworld --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 5e-6 --nu 0.7 > cascade_alfworld.log 2>&1 &

# ALFWorld + ReAct
# Please configure the <YOUR_ALFWORLD_PATH> as the environment pathenv/alfworld-react.py
python -u main.py --env alfworld-react --agent zero-shot --llm qwen3-32b > zero-shot_alfworld-react.log 2>&1 &
python -u main.py --env alfworld-react --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_alfworld-react.log 2>&1 &
python -u main.py --env alfworld-react --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 5e-6 --nu 0.6 > cascade_alfworld-react.log 2>&1 &

# ScienceWorld
python -u main.py --env scienceworld --agent zero-shot --llm qwen3-32b > zero-shot_scienceworld.log 2>&1 &
python -u main.py --env scienceworld --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_scienceworld.log 2>&1 &
python -u main.py --env scienceworld --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.2 > cascade_scienceworld.log 2>&1 &

# ScienceWorld + ReAct
python -u main.py --env scienceworld-react --agent zero-shot --llm qwen3-32b > zero-shot_scienceworld-react.log 2>&1 &
python -u main.py --env scienceworld-react --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_scienceworld-react.log 2>&1 &
python -u main.py --env scienceworld-react --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 2e-5 --nu 0.4 > cascade_scienceworld-react.log 2>&1 &