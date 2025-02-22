export CUDA_VISIBLE_DEVICES=0,1,2,3
DISABLE_VERSION_CHECK=1 llamafactory-cli train /home/liyu/workspaces/llmHMER/LLaMA-Factory/examples/inference/qwen2_vl2B_full_zeroshot.yaml
DISABLE_VERSION_CHECK=1 llamafactory-cli train /home/liyu/workspaces/llmHMER/LLaMA-Factory/examples/inference/qwen2_vl7B_full_zeroshot.yaml
# DISABLE_VERSION_CHECK=1 llamafactory-cli train /home/liyu/workspaces/llmHMER/LLaMA-Factory/examples/inference/qwen2.5_vl3B_full_zeroshot.yaml
# DISABLE_VERSION_CHECK=1 llamafactory-cli train /home/liyu/workspaces/llmHMER/LLaMA-Factory/examples/inference/qwen2.5_vl7B_full_zeroshot.yaml
