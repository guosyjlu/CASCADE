nohup vllm serve <MODEL_PATH> \
    --tokenizer <MODEL_PATH> \
    --served-model-name qwen3-32b \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16000 \
    --max-num-seqs 32 \
    --enforce-eager \
    > vllm_qwen3_32b.log 2>&1 &