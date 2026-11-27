#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export YRCACHE_CONFIG_FILE=/test/workdir/chenn/yrcache/case_full_hit_dram/yrcache.yaml
export YRCACHE_LOG_LEVEL=INFO

# 每次启动使用独立的指标目录
export PROMETHEUS_MULTIPROC_DIR="/test/yrcache_export/tp2_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S)_$$"

mkdir -p "$PROMETHEUS_MULTIPROC_DIR" \
    /test/data_disk/yrcache_dram \
    /test/data_disk_1/yrcache_dram

nohup /test/venv/bin/vllm serve \
    /workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --max-model-len auto \
    --port 8001 \
    --tensor-parallel-size 2 \
    --kv-transfer-config '{"kv_connector":"YRCacheConnector","kv_role":"kv_both","kv_connector_module_path":"yrcache.integration.vllm.yrcache_connector"}' \
    --served-model-name deepseek-14b \
    --trust-remote-code \
    --disable-custom-all-reduce \
    >"/test/workdir/chenn/yrcache/case_full_hit_dram/vllm_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S)_$$.log" \
    2>&1 < /dev/null &
