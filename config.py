import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # Hyper-parameters for the setting
    parser.add_argument("--seed", type=int,
                        default=-1, help="random seed; if set to -1, it would iterate five repeats")
    parser.add_argument("--env", type=str,
                        default="ddxplus", help="Environment type")
    parser.add_argument("--agent", type=str,
                        default="cbr", help="Agent type",
                        choices=["zero-shot", "cbr", "cbrgt", "cbrdiscover", "icrl", "icrl-plus", "reinforce"])                             
    parser.add_argument("--comk", type=int,
                        default=1, help="the number of the retrieved cases in CBR Reuse step")
    parser.add_argument("--recall_size", type=int,
                        default=32, help="the number of the recalled cases in first-stage retrieval")
    
    # Hyper-parameters for LLM
    parser.add_argument("--llm", type=str,
                        default="qwen3-30b-a3b", help="LLM model")
    parser.add_argument("--serving_mode", type=str,
                        default="vllm", help="LLM serving mode: API or vLLM",
                        choices=["openai", "vllm"])
    parser.add_argument("--model_path", type=str,
                        default="<YOUR_MODEL_CHECKPOINT_PATH>", help="LLM model path")  # This arg is only for REINFORCE
    parser.add_argument("--server", type=str,
                        default="localhost", help="local LLM host")
    parser.add_argument("--port", type=int,
                        default=8000, help="server port")        
    parser.add_argument("--max_new_tokens", type=int,
                        default=2048, help="the maximum number of the new tokens")  
    
    # Hyper-parameters for the bandit
    parser.add_argument("--bandit", type=str, 
                        default="NeuralLinLogUCB", help="the algorithm for the bandit policy",
                        choices=["Random", "Pretrained", "LinLogUCB", "NeuralLinUCB",
                            "NeuralLinLogUCB", "NeuralLogUCB"])
    parser.add_argument("--embedding_model_path", type=str,
                        default="<YOUR_MODEL_CHECKPOINT_PATH>", help="Embedding model path,") 
    parser.add_argument("--reranker_model_path", type=str,
                        default="<YOUR_MODEL_CHECKPOINT_PATH>", help="Reranker model path")         
    parser.add_argument("--lamdba", type=float,
                        default=0.1, help="the regularization weight for linear module")                           
    parser.add_argument("--nu", type=float,
                        default=0.1, help="the weight for exploration term")     
    
    # Hyper-parameter for training BERT                                                   
    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size for training NNs")                              
    parser.add_argument("--learning_rate", type=float, 
                        default=2e-5, help="learning rate for training NNs")                              
    parser.add_argument("--kl_coef", type=float, 
                        default=0.1, help="KL coefficient for REINFORCE agent")                               
    parser.add_argument("--weight_decay", type=float, 
                        default=1e-5, help="weight decay for training NNs")
    
    # Hyper-parameters for discovery mechanism
    parser.add_argument("--discover_strategy", type=str,
                        default="random", help="the strategy for case discovery",
                        choices=["random", "ucb", "explore", "exploit"])
    parser.add_argument("--budget_ratio", type=float,
                        default=0.1, help="the ratio of the discover budget to the total interaction rounds")
    parser.add_argument("--queue_size", type=int,
                        default=32, help="the size of sliding window")
    parser.add_argument("--skip_steps", type=int,
                        default=16, help="the steps to skip before each discover decision")
    
    # Hyper-parameters for deepresearch
    parser.add_argument("--web_search", action="store_true", 
                        default=False, help="enable online web search in deepresearcher env")
    args = parser.parse_args()
    return args