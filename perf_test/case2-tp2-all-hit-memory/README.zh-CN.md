# case2-tp2-all-hit-memory ： TP=2 内存 KV 缓存命中性能测试

比较 LMCache 和 YRCache 在 TP=2 下从主机内存回读 KV 缓存的性能。两种后端均配置150GiB内存。默认测试 4K 到 127K 六个输入长度，共 22 个正式并发档，业务输出长度为 1 token。

两个后端分别启动服务后，可以直接运行脚本跑全量测试，也可以使用参数 --scene 32k 单跑 32k场景的所有并发测试。

## 1. 环境和目录

| 组件        | 原测试版本                        |
| --------- | ---------------------------- |
| LMCache   | 0.5.3，使用带 I/O metrics 的定制源码  |
| YRCache   | 1.9.2                        |
| vLLM      | 0.24.0                       |
| EvalScope | 1.10.0                       |
| Python    | 3.12                         |
| 模型        | DeepSeek-R1-Distill-Qwen-14B |

压测脚本按 150 GiB 的内存缓存和 278,880 tokens 的 GPU KV 容量规划，单 token KV 按 192 KiB 估算。实际 GPU KV 容量以 vLLM 启动日志为准。

```text
case2-tp2-all-hit-memory/
├── README.zh-CN.md
├── evalscope/
│   └── kv_memory_prestress_test_tp2.sh
├── lmcache/
│   ├── start_lmcache_tp2.sh
│   └── start_lmcache_vllm_tp2.sh
└── yrcache/
    ├── start_yrcache_vllm_tp2.sh
    └── yrcache.yaml
```

## 2. 测试前需要修改的配置

下文示例的服务 IP 为 `192.168.255.85`。按实际部署替换示例中的 IP 和目录路径。所有用到 `CASE_DIR` 的终端都先执行以下命令。

```bash
export CASE_DIR=/workspaces/work_scripts/perf_test/case2-tp2-all-hit-memory
```

### 2.1 通用配置

| 需要核对的内容       | 修改位置和默认值                                                                                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python 虚拟环境   | 固定使用 `/test/venv`。每个运行启动或压测脚本的终端都先执行 `source /test/venv/bin/activate`，激活成功后再执行其余命令。                                                                                           |
| 模型与 tokenizer | 两个 vLLM 脚本中的模型目录和 EvalScope 的 `TOKENIZER_PATH` 应匹配。默认模型为 `/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`，服务名与 `MODEL` 均为 `deepseek-14b`。 |
| GPU 与上下文      | `CUDA_VISIBLE_DEVICES=0,1`、`--tensor-parallel-size 2`、`--max-model-len auto`。127K 场景需要能处理 130,048 个输入 tokens 及输出。                                                             |
| API 地址        | LMCache 默认端口 `8000`，YRCache 为 `8001`。压测脚本实际默认三个地址均为 `8001`。下文使用命令行参数明确指定三个阶段的地址。                                                                                              |

预热、预埋、驱逐和正式回放须指向同一个推理实例和缓存池。测试期间独占该实例，请在完成整个流程后再停止服务。

### 2.2 LMCache 配置

| 文件                                  | 需要核对或修改的内容                                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lmcache/start_lmcache_tp2.sh`      | L1 为 150 GiB，使用 LRU，驱逐水位和比例为 `0.95`、`0.05`。L2 配置为 `fs_native`，目录为 `/test/data_disk/lmcache_dram`。                                                       |
| `lmcache/start_lmcache_vllm_tp2.sh` | `LMCacheMPConnector` 的 `lmcache.mp.host` 和 `lmcache.mp.port` 应与 server 的 `127.0.0.1:5555` 对应。确保脚本所在目录、`/test/lmcache_logs` 和 `/test/lmcache_export` 可写。 |

### 2.3 YRCache 配置

| 文件                                  | 需要核对或修改的内容                                                                                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yrcache/start_yrcache_vllm_tp2.sh` | 将 `YRCACHE_CONFIG_FILE` 指向本 case 的 `yrcache/yrcache.yaml`，并将脚本末尾的日志重定向目录改为实际可写路径。指标目录默认在 `/test/yrcache_export` 下，每次启动单独创建。                                           |
| `yrcache/yrcache.yaml`              | CPU 缓存开启，`max_cpu_memory_size: 87`、`reserve_memory_size: 12`。本地盘缓存开启，默认目录为 `/test/data_disk/yrcache_dram` 与 `/test/data_disk_1/yrcache_dram`，各有 `:100` 容量后缀。共享存储缓存关闭。 |

如调整 YRCache 的缓存目录，同步修改启动脚本的 `mkdir -p` 路径和 YAML。两种后端均配置了盘缓存，正式请求的回读来源以分层命中统计为准。

### 2.4 场景与并发

| 场景   | 输入 tokens | 每档请求数 | Seed | 正式并发档          |
| ---- | ---------:| -----:| ----:| -------------- |
| 4K   | 4,096     | 64    | 40   | 1、4、8、16、32、64 |
| 8K   | 8,192     | 32    | 41   | 1、4、8、16、32    |
| 16K  | 16,384    | 16    | 42   | 1、4、8、16       |
| 32K  | 32,768    | 8     | 43   | 1、4、8          |
| 64K  | 65,536    | 4     | 44   | 1、4            |
| 127K | 130,048   | 2     | 45   | 1、2            |

预热使用 2 个 8K 请求，2 并发，seed 为 9831。每个场景只预埋一次，预埋使用该场景的最高并发。每个正式并发档前都复用同一组驱逐请求，参数为 **43 个 8,192-token 请求、1 并发、输出 1 token、seed 9841**，合计输入 352,256 tokens。确认实际 GPU KV 容量和命中统计支持驱逐效果，并确保主机内存可同时保留当前场景和驱逐集的 KV。

## 3. LMCache 测试步骤

### 3.1 启动 LMCache 和 vLLM

为本次测试准备独立的缓存目录。终端 A 启动 LMCache，等待 server 就绪。

```bash
source /test/venv/bin/activate
mkdir -p /test/data_disk/lmcache_dram
bash "$CASE_DIR/lmcache/start_lmcache_tp2.sh"
```

终端 B 启动 vLLM，日志同时写入 `lmcache/vllm_lmcache.log`。两个进程均在各自的终端前台运行。

```bash
source /test/venv/bin/activate
bash "$CASE_DIR/lmcache/start_lmcache_vllm_tp2.sh"
```

### 3.2 执行压测

在终端 C 确认模型列表包含 `deepseek-14b`，再执行完整流程。

```bash
source /test/venv/bin/activate
curl -f http://192.168.255.85:8000/v1/models
mkdir -p "$CASE_DIR/results/lmcache-memory-hit"
cd "$CASE_DIR/results/lmcache-memory-hit"
API_URL=http://192.168.255.85:8000/v1/chat/completions
bash "$CASE_DIR/evalscope/kv_memory_prestress_test_tp2.sh" \
  --warmup-url "$API_URL" --prestress-url "$API_URL" --test-url "$API_URL"
```

保存服务日志和结果后，先在 B 停止 vLLM，再在 A 停止 LMCache，确认本次 GPU worker 已退出。

### 3.3 单场景测试

只测某个场景时，可在命令末尾加 `--scene 32k` 等参数。

```bash
source /test/venv/bin/activate
curl -f http://192.168.255.85:8000/v1/models
mkdir -p "\$CASE_DIR/results/lmcache-memory-hit"
cd "\$CASE_DIR/results/lmcache-memory-hit"
API_URL=http://192.168.255.85:8000/v1/chat/completions
bash "\$CASE_DIR/evalscope/kv_memory_prestress_test_tp2.sh" \
  --warmup-url "$API_URL" --prestress-url "$API_URL" --test-url "\$API_URL" --scene 32k
```

## 4. YRCache 测试步骤

### 4.1 启动 YRCache + vLLM

```bash
source /test/venv/bin/activate
bash "$CASE_DIR/yrcache/start_yrcache_vllm_tp2.sh"
```

脚本通过 `YRCacheConnector` 后台启动 vLLM。查看脚本指定目录下的 `vllm_时间戳_PID.log`，等 API 返回包含 `deepseek-14b` 的模型列表。

### 4.2 执行压测

```bash
source /test/venv/bin/activate
curl -f http://192.168.255.85:8001/v1/models
mkdir -p "$CASE_DIR/results/yrcache-memory-hit"
cd "$CASE_DIR/results/yrcache-memory-hit"
API_URL=http://192.168.255.85:8001/v1/chat/completions
bash "$CASE_DIR/evalscope/kv_memory_prestress_test_tp2.sh" \
  --warmup-url "$API_URL" --prestress-url "$API_URL" --test-url "$API_URL"
```

测试结束后，使用 `kill -TERM PID` 中止vLLM进程，等待 worker 退出并用 `nvidia-smi` 检查 GPU。

## 5. 结果与注意事项

每次运行在当前目录追加 `kv_memory_cache_result.csv`，在 `results/kv_memory_cache_hit_test_*/` 下生成 `summary.txt`、`prestress_params.txt`、各阶段日志和 EvalScope 原始报告。复测使用新的结果目录，避免 CSV 混入旧数据。

- 按相同的场景、并发和请求数比较两后端 `phase=test` 行的平均 TTFT、平均 TPOT、输出吞吐和总吞吐。`prestress` 和 `eviction` 行用于核对预埋及驱逐过程。
- `PASS` 表示 EvalScope 命令成功。结合请求成功数、GPU 前缀命中、后端内存命中、SSD 读取和 miss 统计，确认正式请求从主机内存回读。
- `--only-prestress` 可只执行预热和预埋。单独回放时，确认选中的场景 KV 仍在同一实例的内存中，再使用 `--skip-prestress`，并保持模型、tokenizer、请求数和 seed 一致。
