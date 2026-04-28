CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve <MODEL_PATH> \
    --tokenizer <MODEL_PATH> \
    --served-model-name qwen3-14b \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --enforce-eager \
    > vllm_qwen3_14b.log 2>&1 &