CUDA_VISIBLE_DEVICES=0 nohup vllm serve <MODEL_PATH> \
    --tokenizer <MODEL_PATH> \
    --served-model-name qwen3-4b \
    --dtype bfloat16 \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enforce-eager \
    > vllm_qwen3_4b.log 2>&1 &