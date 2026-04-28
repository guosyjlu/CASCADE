nohup vllm serve <MODEL_PATH> \
    --tokenizer <MODEL_PATH> \
    --served-model-name qwen3-30b-a3b \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --max-model-len 24000 \
    --max-num-seqs 32 \
    --enforce-eager \
    > vllm_qwen3_30b_a3b.log 2>&1 &