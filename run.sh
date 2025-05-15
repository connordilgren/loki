# 1. pick GPU 0 only
export CUDA_VISIBLE_DEVICES=2,3

# 2. serialize launches to catch the real error
export CUDA_LAUNCH_BLOCKING=1

# 3. disable allocator warmup
export TRANSFORMERS_NO_CUDA_ALLOCATOR_WARMUP=1

# 4. run your eval (you may still need to edit evaluate_tasks.py
#    to pass device_map/low_cpu_mem_usage/offload_folder into from_pretrained)
CUDA_VISIBLE_DEVICES=5,3 python -u evaluate_tasks.py \
  --sequence-length 2048 \
  --model-id meta-llama/Llama-2-7b-hf \
  --model-type llama \
  --use-pca-topk-sparse \
  --top-r 32 \
  --top-k 0.25 \
  --rotary-type prerotary \
  --dataset wikitext-test \
  --transform-dataset wikitext \
  --use-wandb 
