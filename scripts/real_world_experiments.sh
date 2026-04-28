# This file maintains the Python scripts for the main experiments in multi-turn, real-world results (Fig. 6)

# Deep Search (2Wiki)
# Please configure the OpenAI API interface (API_KEY and BASE_URL)
# Local RAG
# Please serve the local RAG engine before performing DTL.
nohup python -u env/deepsearch/rag_server.py --faiss_gpu > rag_server.log 2>&1 &
python -u main_deepsearch.py --env 2wiki --agent zero-shot --llm qwen3-30b-a3b > zero-shot_2wiki_local_rag.log 2>&1 &
python -u main_deepsearch.py --env 2wiki --agent cbr --bandit Pretrained --llm qwen3-30b-a3b > npcbr_2wiki_local_rag.log 2>&1 &
python -u main_deepsearch.py --env 2wiki --agent cbr --bandit NeuralLinLogUCB --llm qwen3-30b-a3b --learning_rate 2e-5 --nu 0.2 > cascade_2wiki_local_rag.log 2>&1 &
# Web Search
# Please configure <YOUR_SERPER_API_KEY> in env/deepsearch/serper_snippert_mcp.py
python -u main_deepsearch.py --env 2wiki --web_search --agent zero-shot --llm qwen3-30b-a3b > zero-shot_2wiki_web_search.log 2>&1 &
python -u main_deepsearch.py --env 2wiki --web_search --agent cbr --bandit Pretrained --llm qwen3-30b-a3b > npcbr_2wiki_web_search.log 2>&1 &
python -u main_deepsearch.py --env 2wiki --web_search --agent cbr --bandit NeuralLinLogUCB --llm qwen3-30b-a3b --learning_rate 2e-5 --nu 0.1 > cascade_2wiki_web_search.log 2>&1 &

# MIMIC-III (EHR)
# Please configure <CASCADE_DIR_PATH> as the firectory path in env/ehragent/tabtools.py
python -u main.py --env ehr --agent zero-shot --llm qwen3-32b > zero-shot_ehr.log 2>&1 &
python -u main.py --env ehr --agent cbr --bandit Pretrained --llm qwen3-32b > npcbr_ehr.log 2>&1 &
python -u main.py --env ehr --agent cbr --bandit NeuralLinLogUCB --llm qwen3-32b --learning_rate 1e-5 --nu 0.7 > cascade_ehr.log 2>&1 &
