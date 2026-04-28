# This file maintains the Python scripts for black-box models in single-turn results (Fig. 4b)

## gemini-2.0-flash

# 1. Zero-shot
python -u main.py --env ddxplus --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_ddxplus_gemini_2.0_flash.log 2>&1 &
python -u main.py --env mud --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_mud_gemini_2.0_flash.log 2>&1 &
python -u main.py --env cmdl --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_cmdl_gemini_2.0_flash.log 2>&1 &
python -u main.py --env banking77 --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_banking77_gemini_2.0_flash.log 2>&1 &
python -u main.py --env sentifin --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_sentifin_gemini_2.0_flash.log 2>&1 &
python -u main.py --env rca --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_rca_gemini_2.0_flash.log 2>&1 &
python -u main.py --env lfd --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_lfd_gemini_2.0_flash.log 2>&1 &
python -u main.py --env spider --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_spider_gemini_2.0_flash.log 2>&1 &
python -u main.py --env bird --agent zero-shot --llm gemini-2.0-flash --serving_mode openai > zero-shot_bird_gemini_2.0_flash.log 2>&1 &

# 2. NP-CBR
python -u main.py --env ddxplus --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_ddxplus_gemini_2.0_flash.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_mud_gemini_2.0_flash.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_cmdl_gemini_2.0_flash.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_banking77_gemini_2.0_flash.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_sentifin_gemini_2.0_flash.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_rca_gemini_2.0_flash.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_lfd_gemini_2.0_flash.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_spider_gemini_2.0_flash.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit Pretrained --llm gemini-2.0-flash --serving_mode openai > npcbr_bird_gemini_2.0_flash.log 2>&1 &

# 3. CASCADE
python -u main.py --env ddxplus --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 1e-5 --nu 0.2 > cascade_ddxplus_gemini_2.0_flash.log 2>&1 &
python -u main.py --env mud --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 5e-6 --nu 0.2 > cascade_mud_gemini_2.0_flash.log 2>&1 &
python -u main.py --env cmdl --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 5e-6 --nu 0.1 > cascade_cmdl_gemini_2.0_flash.log 2>&1 &
python -u main.py --env banking77 --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 1e-5 --nu 0.1 > cascade_banking77_gemini_2.0_flash.log 2>&1 &
python -u main.py --env sentifin --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 1e-5 --nu 0.4 > cascade_sentifin_gemini_2.0_flash.log 2>&1 &
python -u main.py --env rca --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 2e-5 --nu 1.0 > cascade_rca_gemini_2.0_flash.log 2>&1 &
python -u main.py --env lfd --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 2e-5 --nu 0.4 > cascade_lfd_gemini_2.0_flash.log 2>&1 &
python -u main.py --env spider --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 1e-5 --nu 0.8 > cascade_spider_gemini_2.0_flash.log 2>&1 &
python -u main.py --env bird --agent cbr --bandit NeuralLinLogUCB --llm gemini-2.0-flash --serving_mode openai --learning_rate 5e-6 --nu 0.1 > cascade_bird_gemini_2.0_flash.log 2>&1 &
