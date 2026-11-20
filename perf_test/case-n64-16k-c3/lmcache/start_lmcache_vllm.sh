export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PROMETHEUS_MULTIPROC_DIR=/mooncake_test/lmcache_export

mkdir -p /mooncake_test/lmcache_logs /mooncake_test/lmcache_export

/mooncake_test/venv/bin/vllm serve \
  /workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --max-model-len auto \
  --port 8000 \
  --served-model-name deepseek-14b \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "127.0.0.1",
      "lmcache.mp.port": 5555,
      "lmcache.mp.mp_transfer_mode": "lmcache_driven"
    }
  }'
