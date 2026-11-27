# case4-tp2-all-hit-ssd-bio · TP=2 SSD BIO KV 缓存命中性能测试

比较 LMCache 和 YRCache 在 TP=2 下使用本地 SSD（Buffered I/O）回读 KV 缓存的性能。测试覆盖 4K 到 127K 六个输入长度，业务输出长度默认为 1 token。

两种后端分别启动服务后，可以直接运行完整测试矩阵，也可以参考快捷脚本单独执行不同场景不同并发的测试。

## 1. 环境和目录

| 组件        | 原测试版本                        |
| --------- | ---------------------------- |
| LMCache   | 0.5.3，使用带 I/O metrics 的定制源码  |
| YRCache   | 1.9.0                        |
| vLLM      | 0.24.0                       |
| EvalScope | 1.10.0                       |
| Python    | 3.12                         |
| 模型        | DeepSeek-R1-Distill-Qwen-14B |

```text
case4-tp2-all-hit-ssd-bio/
├── README.zh-CN.md
├── evalscope/
│   ├── tp2_kv_prestress_test.sh   # 完整矩阵及单档测试
│   └── test_32k_64k_127k.sh       # 长输入快捷入口
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
export CASE_DIR=/workspaces/work_scripts/perf_test/case4-tp2-all-hit-ssd-bio
```

### 2.1 通用配置

| 需要核对的内容       | 修改位置和默认值                                                                                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python 虚拟环境   | 固定使用 `/test/venv`。每个运行启动或压测脚本的终端都先执行 `source /test/venv/bin/activate`，激活成功后再执行其余命令。                                                                                           |
| 模型与 tokenizer | 两个 vLLM 脚本中的模型目录和 EvalScope 的 `TOKENIZER_PATH` 应匹配。默认模型为 `/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`，服务名与 `MODEL` 均为 `deepseek-14b`。 |
| GPU 与上下文      | `CUDA_VISIBLE_DEVICES=0,1`、`--tensor-parallel-size 2`、`--max-model-len auto`。127K 场景需要能处理 130,048 个输入 tokens 及输出。                                                             |
| API 地址        | LMCache 默认端口 `8000`，YRCache 为 `8001`。`tp2_kv_prestress_test.sh` 实际默认三个地址均为 `8000`。直接运行时可通过 `--warmup-url`、`--prestress-url`、`--test-url` 覆盖。                                  |

测试前先停止上一轮的服务，为本次填充准备空的专用缓存目录。压测期间独占推理实例。需要在预埋和回放之间重启时，保留预埋写入的数据。

### 2.2 LMCache 配置

| 文件                                  | 需要核对或修改的内容                                                                                                                                                                                            |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lmcache/start_lmcache_tp2.sh`      | L1 为 150 GiB，使用 LRU，驱逐水位和比例为 `0.6`、`0.2`。L2 写入策略为 `skip_l1`。两组 `fs_native` 适配器的 `use_odirect` 均为 `false`，持久化开启。`base_path` 分别为 `/test/data_disk/lmcache_ssd_bio`、`/test/data_disk_1/lmcache_ssd_bio`。 |
| `lmcache/start_lmcache_vllm_tp2.sh` | `LMCacheMPConnector` 的 `lmcache.mp.host` 和 `lmcache.mp.port` 应与 server 的 `127.0.0.1:5555` 对应。确保脚本所在目录和 `PROMETHEUS_MULTIPROC_DIR` 可写。                                                                 |

### 2.3 YRCache 配置

| 文件                                  | 需要核对或修改的内容                                                                                                                                                                                                      |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yrcache/start_yrcache_vllm_tp2.sh` | 将 `YRCACHE_CONFIG_FILE` 指向本 case 的 `yrcache/yrcache.yaml`，并将脚本末尾的日志重定向目录改为实际可写路径。指标目录默认在 `/test/yrcache_export` 下，每次启动单独创建。                                                                                     |
| `yrcache/yrcache.yaml`              | CPU 缓存开启，`max_cpu_memory_size: 75`、`reserve_memory_size: 35`。本地盘缓存开启，`use_direct_io: false`，共享存储缓存关闭。`local_disks` 默认指向 `/test/data_disk/yrcache_ssd_bio` 和 `/test/data_disk_1/yrcache_ssd_bio`，各有 `:100` 容量后缀。 |

如调整 YRCache 的 SSD 路径，同步修改启动脚本的 `mkdir -p` 路径和 YAML。两种后端各自使用独立的专用缓存目录。

### 2.4 场景与并发

| 场景   | 输入 tokens | 基准请求数 | Seed | 预埋并发 | 正式并发档          |
| ---- | ---------:| -----:| ----:| ----:| -------------- |
| 4K   | 4,096     | 64    | 40   | 64   | 1、4、8、16、32、64 |
| 8K   | 8,192     | 32    | 41   | 32   | 1、4、8、16、32    |
| 16K  | 16,384    | 16    | 42   | 16   | 1、4、8、16       |
| 32K  | 32,768    | 8     | 43   | 8    | 1、4、8          |
| 64K  | 65,536    | 4     | 44   | 4    | 1、4            |
| 127K | 130,048   | 2     | 45   | 2    | 1              |

完整矩阵先用 10 个 8K 请求预热（2 并发，seed 9831），然后按各场景最高并发预埋。回放按 6 轮穿插，共 31 步，其中 21 个为正式档，10 个垫场复用相应场景的请求做缓存置换。

可用 `--scene 32k` 等参数限定场景。`--skip-padding` 会改变用于置换缓存的请求顺序，比较完整矩阵时两种后端应使用相同的选项。

## 3. LMCache 测试步骤

### 3.1 启动 LMCache 和 vLLM

在两处 SSD 挂载点准备目录。终端 A 启动 LMCache server 并保持运行。

```bash
source /test/venv/bin/activate
mkdir -p /test/data_disk/lmcache_ssd_bio /test/data_disk_1/lmcache_ssd_bio
bash "$CASE_DIR/lmcache/start_lmcache_tp2.sh"
```

终端 B 先准备指标目录，再前台启动 vLLM。日志同时写入 `lmcache/vllm_lmcache.log`。

```bash
source /test/venv/bin/activate
mkdir -p /test/lmcache_export
bash "$CASE_DIR/lmcache/start_lmcache_vllm_tp2.sh"
```

### 3.2 执行完整矩阵

在终端 C 确认模型列表包含 `deepseek-14b`，再运行压测。以下命令采用底层脚本默认的基准请求数。

```bash
source /test/venv/bin/activate
curl -f http://192.168.255.85:8000/v1/models
mkdir -p "$CASE_DIR/results/lmcache-ssd-bio"
cd "$CASE_DIR/results/lmcache-ssd-bio"
API_URL=http://192.168.255.85:8000/v1/chat/completions
bash "$CASE_DIR/evalscope/tp2_kv_prestress_test.sh" \
  --warmup-url "$API_URL" --prestress-url "$API_URL" --test-url "$API_URL" --skip-padding --request-multiplier 5
```

保存日志与结果后，在终端 B 停止 vLLM，确认 worker 退出，再停止终端 A 的 LMCache server。

## 4. YRCache 测试步骤

### 4.1 启动 YRCache + vLLM

```bash
source /test/venv/bin/activate
bash "$CASE_DIR/yrcache/start_yrcache_vllm_tp2.sh"
```

脚本通过 `YRCacheConnector` 后台启动 vLLM。查看脚本指定目录下的 `vllm_时间戳_PID.log`，等 API 返回包含 `deepseek-14b` 的模型列表。

### 4.2 执行完整矩阵

```bash
source /test/venv/bin/activate
curl -f http://192.168.255.85:8001/v1/models
mkdir -p "$CASE_DIR/results/yrcache-ssd-bio"
cd "$CASE_DIR/results/yrcache-ssd-bio"
API_URL=http://192.168.255.85:8001/v1/chat/completions
bash "$CASE_DIR/evalscope/tp2_kv_prestress_test.sh" \
  --warmup-url "$API_URL" --prestress-url "$API_URL" --test-url "$API_URL" --skip-padding --request-multiplier 5 
```

测试结束后，使用 `kill -TERM PID` 中止vLLM进程，等待 worker 退出并用 `nvidia-smi` 检查 GPU。

## 5. 结果与注意事项

### 5.1 只测三个长输入场景

`test_32k_64k_127k.sh` 先依次预埋 32K、64K、127K，只在第一次预埋前执行预热。随后等待 90 秒，在同一个实例上按原编号 16、11、12 回放。该入口通过 `--request-multiplier 5` 将预埋和回放的请求数放大到基准的 5 倍，并发不变。

| 场景   | 请求数 | 并发  | 原用例编号         |
| ---- | ---:| ---:| ------------- |
| 32K  | 40  | 8   | 16            |
| 64K  | 20  | 4   | 11            |
| 127K | 10  | 2   | 12（完整矩阵中的垫场项） |

运行快捷脚本前，将 `tp2_kv_prestress_test.sh` 中的 `WARMUP_URL`、`PRESTRESS_URL`、`TEST_URL` 都设为当前后端的地址。快捷脚本直接使用这些默认值，并在自身所在的 `evalscope/` 目录写入 CSV 和 `results/`。切换后端前归档本轮结果。

```bash
source /test/venv/bin/activate
bash "$CASE_DIR/evalscope/test_32k_64k_127k.sh"
```

### 5.2 核对层级命中和结果

- `PASS` 表示 EvalScope 命令成功。结合实际成功请求数、GPU 和 CPU 层命中、SSD 层读取和 miss 日志核对回读来源。Buffered I/O 的 SSD 层命中可能由操作系统页缓存提供。服务重启后页缓存仍可能保留，结合后端分层统计和 SSD 设备读量判断实际读盘情况。
- 如需清除应用内存和 GPU 中的 KV，先用 `--only-prestress` 预埋，确认写入或 checkpoint 完成后停止服务，保留盘数据并重启。用 `/v1/models` 检查就绪，然后加 `--skip-prestress --skip-warmup` 回放，保持场景、请求数倍数、模型和 tokenizer 一致。可用 `--case N` 按原编号测量单档。
- 主脚本在当前目录追加 `kv_cache_result.csv`，详细数据位于 `results/kv_cache_hit_test_*/`，包括 `summary.txt`、`prestress_params.txt`、各阶段日志及 EvalScope 原始报告。复测使用新的结果目录。
- 比较两后端相同编号、请求数和并发的 `phase=test` 指标，记录平均 TTFT、平均 TPOT、输出吞吐和总吞吐。对完整矩阵按原编号识别垫场，长输入快捷结果保留其实际并发及请求数。
