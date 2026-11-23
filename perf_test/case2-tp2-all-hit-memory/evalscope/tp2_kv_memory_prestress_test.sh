#!/bin/bash
set -o pipefail

# 运行前提:
#   1. 内存缓存池按 150 GiB 规划, 默认按 4K -> 8K -> 16K -> 32K -> 64K -> 127K 逐场景执行。
#   2. GPU KV Cache 可缓存 278,880 tokens; 两卡每卡约 25.53 GiB, 单 token KV 为 192 KiB。
#   3. 预热、预埋和测试指向同一推理实例及缓存池, 测试期间无其他业务请求干扰。
#   4. 六个场景使用不同请求内容, 各场景预埋与测试使用相同 seed/number。
#   5. 预埋数据需已写入内存; 当前场景 KV 和固定驱逐集需保留在内存, 已结束场景的数据允许淘汰。
# 每个场景只预埋一次, 每个正式并发档前使用同一组固定请求驱逐 GPU 缓存。
# 每次驱逐: 43条8192 tokens, 1并发, 输出1 token, seed9841, 累计输入352,256 tokens。
# PASS 沿用原脚本的 evalscope 退出码判断; 是否从内存读回需结合缓存日志确认。

#--------------------------default-----------------------------------
# 预热/预埋走 8000, 缓存命中测试走 8000
WARMUP_URL="http://192.168.255.85:8000/v1/chat/completions"
PRESTRESS_URL="http://192.168.255.85:8000/v1/chat/completions"
TEST_URL="http://192.168.255.85:8000/v1/chat/completions"
MODEL="deepseek-14b"
TOKENIZER_PATH="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

MIN_TOKENS=1
MAX_TOKENS=1

# 预热: 重启模型后组件加载, 2并发2个8k请求
WARMUP_PARALLEL=2
WARMUP_NUMBER=2
WARMUP_PROMPT=8192
WARMUP_SEED=9831

# GPU 缓存驱逐: 每次复用同一组请求, 不随场景或并发档改变 seed。
EVICT_PARALLEL=1
EVICT_NUMBER=43
EVICT_PROMPT=8192
EVICT_TOKENS=1
EVICT_SEED=9841

# 预埋并发 = 测试矩阵中该场景的最大并发 (从 TEST_LIST 自动推导, 加快预埋)

SKIP_WARMUP=0
SKIP_PRESTRESS=0
ONLY_PRESTRESS=0
SELECTED_SCENE="all"

# 场景: 名称:prompt长度:预埋条数:seed (预埋与测试共用 seed/number)
SCENE_LIST=(
    "4k:4096:64:40"
    "8k:8192:32:41"
    "16k:16384:16:42"
    "32k:32768:8:43"
    "64k:65536:4:44"
    "127k:130048:2:45"
)

# 测试: 场景:并发, 全部六个场景共22个正式用例。
# 当前场景的所有并发档完成后, 再预埋并测试下一个场景。
TEST_LIST=(
    # 4K: 每档64条
    "4k:1"
    "4k:4"
    "4k:8"
    "4k:16"
    "4k:32"
    "4k:64"
    # 8K: 每档32条
    "8k:1"
    "8k:4"
    "8k:8"
    "8k:16"
    "8k:32"
    # 16K: 每档16条
    "16k:1"
    "16k:4"
    "16k:8"
    "16k:16"
    # 32K: 每档8条
    "32k:1"
    "32k:4"
    "32k:8"
    # 64K: 每档4条
    "64k:1"
    "64k:4"
    # 127K: 每档2条
    "127k:1"
    "127k:2"
)

CSV_FILE="./kv_memory_cache_result.csv"

#--------------------------functions-----------------------------------

show_help() {
    cat <<'EOF'
用法: ./kv_memory_prestress_test.sh [选项]

KV Cache 预埋 + 内存回读性能测试
运行前提:
  内存缓存池按 150 GiB 规划, GPU KV Cache 容量为 278,880 tokens, 单 token KV 为 192 KiB。
  预热、预埋和测试指向同一推理实例及缓存池, 测试期间无其他业务请求干扰。
  预埋数据需已写入内存; 当前场景 KV 和固定驱逐集需保留在内存, 已结束场景的数据允许淘汰。
  本脚本不修改服务端配置; PASS 表示 evalscope 命令成功, 回读来源需结合缓存日志确认。

流程:
  1. 模型预热: 2并发 2个 8k 请求 (组件加载)
  2. 默认按 4k -> 8k -> 16k -> 32k -> 64k -> 127k 逐场景执行, 也可使用 --scene 指定单个场景。
  3. 当前场景每个并发档测试前, 先执行一次固定请求驱逐 GPU 缓存:
     43条8192 tokens, 1并发, 输出1 token, seed9841; 每轮输入352,256 tokens。
  4. 当前场景全部并发档完成后, 再进入下一个场景。

场景与并发 (全部场景共22个正式用例, 每个场景只预埋一次, 业务输出默认1 token):
  4k  : 64条, seed40, 预埋64并发; 测试1 / 4 / 8 / 16 / 32 / 64并发
  8k  : 32条, seed41, 预埋32并发; 测试1 / 4 / 8 / 16 / 32并发
  16k : 16条, seed42, 预埋16并发; 测试1 / 4 / 8 / 16并发
  32k : 8条,  seed43, 预埋8并发;  测试1 / 4 / 8并发
  64k : 4条,  seed44, 预埋4并发;  测试1 / 4并发
  127k: 2条,  seed45, 预埋2并发;  测试1 / 2并发 (输入130,048 tokens)

选项:
  --scene NAME          选择场景: all / 4k / 8k / 16k / 32k / 64k / 127k, 默认 all
  --warmup-url URL       预热地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --prestress-url URL    预埋地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --test-url URL         测试地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --model NAME           模型名, 默认 deepseek-14b
  --tokenizer-path PATH  tokenizer 路径
  --min-tokens N         输出 token 下限, 默认 1
  --max-tokens N         输出 token 上限, 默认 1
  --skip-warmup          跳过预热
  --skip-prestress       跳过预埋 (已预埋过时使用)
  --only-prestress       仅预热+预埋, 不跑测试矩阵
  -h, --help             帮助
EOF
}

validate_positive_integer() {
    local value="$1" var_name="${2:-值}"
    value=$(echo "$value" | xargs)
    if [[ -z "$value" ]] || ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: $var_name 必须是大于0的整数, 当前值: '$value'" >&2
        return 1
    fi
    return 0
}

banner() {
    echo ""
    echo "========================================================="
    echo "  $1"
    echo "========================================================="
}

#--------------------------parse parameters--------------------------

while [ $# -gt 0 ]; do
    case $1 in
        --scene)
            SELECTED_SCENE="${2:-}"
            SELECTED_SCENE="${SELECTED_SCENE,,}"
            case "$SELECTED_SCENE" in
                all|4k|8k|16k|32k|64k|127k) shift 2 ;;
                *)
                    echo "错误: --scene 必须指定 all / 4k / 8k / 16k / 32k / 64k / 127k" >&2
                    exit 1
                    ;;
            esac
            ;;
        --warmup-url)       WARMUP_URL="$2"; shift 2 ;;
        --prestress-url)    PRESTRESS_URL="$2"; shift 2 ;;
        --test-url)         TEST_URL="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --tokenizer-path)   TOKENIZER_PATH="$2"; shift 2 ;;
        --min-tokens)       MIN_TOKENS="$2"; shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2"; shift 2 ;;
        --skip-warmup)      SKIP_WARMUP=1; shift ;;
        --skip-prestress)   SKIP_PRESTRESS=1; shift ;;
        --only-prestress)   ONLY_PRESTRESS=1; shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)                  echo "未知选项: $1"; exit 1 ;;
    esac
done

validate_positive_integer "$MIN_TOKENS" "min-tokens" || exit 1
validate_positive_integer "$MAX_TOKENS" "max-tokens" || exit 1

command -v evalscope >/dev/null 2>&1 || { echo "错误: 未找到 evalscope 命令" >&2; exit 1; }

#--------------------------scene--------------------------

# 指定单个场景时, 只保留该场景及其正式用例。
if [[ "$SELECTED_SCENE" != "all" ]]; then
    for entry in "${SCENE_LIST[@]}"; do
        IFS=':' read -r name ptok pnum sseed <<< "$entry"
        if [[ "$name" == "$SELECTED_SCENE" ]]; then
            SCENE_LIST=("$entry")
            break
        fi
    done

    SELECTED_TEST_LIST=()
    for entry in "${TEST_LIST[@]}"; do
        IFS=':' read -r name par <<< "$entry"
        if [[ "$name" == "$SELECTED_SCENE" ]]; then
            SELECTED_TEST_LIST+=("$entry")
        fi
    done
    TEST_LIST=("${SELECTED_TEST_LIST[@]}")
fi

declare -A SCENE_TOKENS SCENE_NUMBER SCENE_SEED SCENE_PARALLEL
SCENE_TAG=""
for entry in "${SCENE_LIST[@]}"; do
    IFS=':' read -r name ptok pnum sseed <<< "$entry"
    SCENE_TOKENS[$name]=$ptok
    SCENE_NUMBER[$name]=$pnum
    SCENE_SEED[$name]=$sseed
    SCENE_TAG="${SCENE_TAG}${name}-"
done
SCENE_TAG="${SCENE_TAG%-}"    # 全部场景或 --scene 指定的单个场景

# 每场景预埋并发 = 测试矩阵中该场景的最大并发
for entry in "${TEST_LIST[@]}"; do
    IFS=':' read -r name par <<< "$entry"
    if [[ -z "${SCENE_PARALLEL[$name]}" ]] || (( par > SCENE_PARALLEL[$name] )); then
        SCENE_PARALLEL[$name]=$par
    fi
done

#--------------------------prepare dirs & logs--------------------------

# 结果目录命名与测试项关联: 测试项_模型_场景_时间
MODEL_TAG=${MODEL//\//-}
RESULT_ROOT="./results/kv_memory_cache_hit_test_${MODEL_TAG}_${SCENE_TAG}_$(date "+%Y%m%d_%H%M%S")"
LOG_DIR="$RESULT_ROOT/logs"

mkdir -p "$LOG_DIR"
SUMMARY_LOG="$RESULT_ROOT/summary.txt"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

RECORD_FILE="$RESULT_ROOT/prestress_params.txt"
{
    echo "# 预埋参数记录 (测试阶段必须使用相同 seed/number)"
    echo "# model=$MODEL"
    echo "# tokenizer=$TOKENIZER_PATH"
    echo "# scene=$SELECTED_SCENE"
    echo "# 运行前提: 内存缓存池按 150 GiB 规划; GPU KV Cache 容量为 278,880 tokens; 单 token KV 为 192 KiB"
    echo "# min_tokens=$MIN_TOKENS max_tokens=$MAX_TOKENS"
    echo "# warmup: parallel=$WARMUP_PARALLEL number=$WARMUP_NUMBER prompt=$WARMUP_PROMPT seed=$WARMUP_SEED"
    echo "# eviction: parallel=$EVICT_PARALLEL number=$EVICT_NUMBER prompt=$EVICT_PROMPT seed=$EVICT_SEED min_tokens=$EVICT_TOKENS max_tokens=$EVICT_TOKENS"
    for entry in "${SCENE_LIST[@]}"; do
        IFS=':' read -r name ptok pnum sseed <<< "$entry"
        printf "%-5s prompt_length=%-7s number=%-5s seed=%-4s prestress_parallel=%s\n" "$name" "$ptok" "$pnum" "$sseed" "${SCENE_PARALLEL[$name]}"
    done
} > "$RECORD_FILE"

if [ ! -f "$CSV_FILE" ]; then
    echo "timestamp,model,phase,scene,case,seed,prompt_length,number,parallel,url,TTFT(ms),TPOT(ms),Output Throughput(tok/s),Total Throughput(tok/s),status" > "$CSV_FILE"
fi

#--------------------------run & collect--------------------------

# 运行一次 evalscope 压测: tag url parallel number prompt_len seed [min_tokens max_tokens]
# 未指定输出长度时使用业务参数; 驱逐请求单独传入 1 / 1。
run_case() {
    local tag=$1 url=$2 parallel=$3 number=$4 ptok=$5 seed=$6
    local min_tokens=${7:-$MIN_TOKENS} max_tokens=${8:-$MAX_TOKENS}
    local outdir="$RESULT_ROOT/$tag"
    mkdir -p "$outdir"
    echo "--- [$tag] start $(date "+%Y-%m-%d %H:%M:%S") (url=$url parallel=$parallel number=$number prompt=$ptok seed=$seed) ---"
    evalscope perf \
        --url "$url" \
        --parallel "$parallel" \
        --model "$MODEL" \
        --number "$number" \
        --api openai \
        --dataset random \
        --outputs-dir "$outdir" \
        --min-prompt-length "$ptok" \
        --max-prompt-length "$ptok" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --min-tokens "$min_tokens" \
        --max-tokens "$max_tokens" \
        --seed "$seed" \
        > "$LOG_DIR/${tag}.log" 2>&1
    local rc=$?
    echo "--- [$tag] end $(date "+%Y-%m-%d %H:%M:%S") (exit=$rc, log=$LOG_DIR/${tag}.log) ---"
    return $rc
}

# 直接提取 evalscope "Benchmarking summary" 表格中的原始值, 不做任何计算/换算
# 输出: TTFT(ms),TPOT(ms),Output Throughput(tok/s),Total Throughput(tok/s)
collect_metrics() {
    local latest_log
    latest_log=$(find "$1" -name 'benchmark.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
    [[ -z "$latest_log" ]] && return 1
    awk -F'│' '
        /Avg TTFT \(ms\)/              { v=$3; gsub(/[[:space:]]/,"",v); ttft=v }
        /Avg TPOT \(ms\)/              { v=$3; gsub(/[[:space:]]/,"",v); tpot=v }
        /Output Throughput \(tok\/s\)/ { v=$3; gsub(/[[:space:]]/,"",v); ot=v }
        /Total Throughput \(tok\/s\)/  { v=$3; gsub(/[[:space:]]/,"",v); tot=v }
        END { if (ttft!="") printf "%s,%s,%s,%s", ttft, tpot, ot, tot }
    ' "$latest_log"
}

# phase scene case seed ptok number parallel url status outdir
append_csv() {
    local metrics
    metrics=$(collect_metrics "${10}")
    [[ -z "$metrics" ]] && metrics=",,,"
    echo "$(date "+%Y-%m-%d %H:%M:%S"),$MODEL,$1,$2,$3,$4,$5,$6,$7,$8,$metrics,$9" >> "$CSV_FILE"
    echo "    指标(evalscope原始值): TTFT/TPOT/OutputThroughput/TotalThroughput = $metrics"
}

# 失败 case 复现脚本: tag url parallel number ptok seed [min_tokens max_tokens]
gen_rerun() {
    local tag=$1 url=$2 parallel=$3 number=$4 ptok=$5 seed=$6
    local min_tokens=${7:-$MIN_TOKENS} max_tokens=${8:-$MAX_TOKENS}
    local f="$RESULT_ROOT/rerun_${tag}.sh"
    cat > "$f" <<EOF
#!/bin/bash
# 复现失败 case: $tag
evalscope perf \\
    --url "$url" \\
    --parallel "$parallel" \\
    --model "$MODEL" \\
    --number "$number" \\
    --api openai \\
    --dataset random \\
    --outputs-dir "$RESULT_ROOT/$tag" \\
    --min-prompt-length "$ptok" \\
    --max-prompt-length "$ptok" \\
    --tokenizer-path "$TOKENIZER_PATH" \\
    --min-tokens "$min_tokens" \\
    --max-tokens "$max_tokens" \\
    --seed "$seed"
EOF
    chmod +x "$f"
    echo "已生成复现脚本: $f"
}

#--------------------------print plan--------------------------

banner "KV Cache 预埋与内存回读性能测试"
echo "Model        : $MODEL"
echo "Tokenizer    : $TOKENIZER_PATH"
echo "预热地址     : $WARMUP_URL"
echo "预埋地址     : $PRESTRESS_URL"
echo "测试地址     : $TEST_URL"
echo "输出 tokens  : $MIN_TOKENS ~ $MAX_TOKENS"
echo "结果目录     : $RESULT_ROOT"
echo "汇总 CSV     : $CSV_FILE"
echo "选择场景     : $SELECTED_SCENE"
echo "内存缓存池   : 150 GiB"
echo "GPU KV Cache : 278,880 tokens (两卡每卡约 25.53 GiB, 单 token KV 为 192 KiB)"
echo "执行方式     : $SCENE_TAG 逐场景预埋和测试, 共${#TEST_LIST[@]}个正式用例"
echo "结果说明     : PASS 表示 evalscope 命令成功; 内存/SSD 回读来源需结合缓存日志确认"
echo ""
echo "预热: parallel=$WARMUP_PARALLEL number=$WARMUP_NUMBER prompt=$WARMUP_PROMPT seed=$WARMUP_SEED"
echo "驱逐: parallel=$EVICT_PARALLEL number=$EVICT_NUMBER prompt=$EVICT_PROMPT seed=$EVICT_SEED output_tokens=$EVICT_TOKENS"
echo ""
echo "场景 (预埋与测试共用 seed/number):"
for entry in "${SCENE_LIST[@]}"; do
    IFS=':' read -r name ptok pnum sseed <<< "$entry"
    echo "  $name : prompt=$ptok, number=$pnum, seed=$sseed, 预埋并发=${SCENE_PARALLEL[$name]}"
done
echo ""
echo "测试矩阵:"
i=0
for entry in "${TEST_LIST[@]}"; do
    i=$((i + 1))
    IFS=':' read -r name par <<< "$entry"
    echo "  $i. $name ${par}并发 (number=${SCENE_NUMBER[$name]}, seed=${SCENE_SEED[$name]})"
done
cat "$RECORD_FILE"

FAIL_COUNT=0

#--------------------------1. warmup--------------------------

if [[ $SKIP_WARMUP -eq 0 ]]; then
    banner "阶段1 模型预热 (组件加载)"
    tag="warmup_8k"
    if run_case "$tag" "$WARMUP_URL" "$WARMUP_PARALLEL" "$WARMUP_NUMBER" "$WARMUP_PROMPT" "$WARMUP_SEED"; then
        echo "✅ [$tag] PASS"
    else
        echo "❌ 预热失败 [$tag], 模型未就绪, 终止 (log=$LOG_DIR/${tag}.log)" >&2
        exit 1
    fi
else
    echo ""
    echo "跳过预热 (--skip-warmup)"
fi

#--------------------------2. per-scene prestress & test--------------------------

ti=0
total=${#TEST_LIST[@]}
for scene_entry in "${SCENE_LIST[@]}"; do
    IFS=':' read -r name ptok pnum sseed <<< "$scene_entry"

    if [[ $SKIP_PRESTRESS -eq 0 ]]; then
        banner "$name 场景预埋一次"
        ppar=${SCENE_PARALLEL[$name]}
        tag="prestress_${name}"
        if run_case "$tag" "$PRESTRESS_URL" "$ppar" "$pnum" "$ptok" "$sseed"; then
            append_csv prestress "$name" 0 "$sseed" "$ptok" "$pnum" "$ppar" "$PRESTRESS_URL" PASS "$RESULT_ROOT/$tag"
            echo "✅ [$tag] PASS"
        else
            append_csv prestress "$name" 0 "$sseed" "$ptok" "$pnum" "$ppar" "$PRESTRESS_URL" FAIL "$RESULT_ROOT/$tag"
            gen_rerun "$tag" "$PRESTRESS_URL" "$ppar" "$pnum" "$ptok" "$sseed"
            echo "❌ 预埋失败 [$tag], 预埋不完整将导致缓存无法命中, 终止" >&2
            exit 1
        fi
    else
        echo ""
        echo "跳过 $name 预埋 (--skip-prestress)"
    fi

    if [[ $ONLY_PRESTRESS -eq 0 ]]; then
        banner "$name 场景内存回读测试 (每个并发档前驱逐 GPU 缓存)"
        for entry in "${TEST_LIST[@]}"; do
            IFS=':' read -r test_scene par <<< "$entry"
            [[ "$test_scene" != "$name" ]] && continue
            ti=$((ti + 1))

            tag=$(printf "evict_%02d_%s_b%s" "$ti" "$name" "$par")
            if run_case "$tag" "$TEST_URL" "$EVICT_PARALLEL" "$EVICT_NUMBER" "$EVICT_PROMPT" "$EVICT_SEED" "$EVICT_TOKENS" "$EVICT_TOKENS"; then
                append_csv eviction "$name" "$ti" "$EVICT_SEED" "$EVICT_PROMPT" "$EVICT_NUMBER" "$EVICT_PARALLEL" "$TEST_URL" PASS "$RESULT_ROOT/$tag"
                echo "✅ [$tag] 驱逐请求完成"
            else
                append_csv eviction "$name" "$ti" "$EVICT_SEED" "$EVICT_PROMPT" "$EVICT_NUMBER" "$EVICT_PARALLEL" "$TEST_URL" FAIL "$RESULT_ROOT/$tag"
                gen_rerun "$tag" "$TEST_URL" "$EVICT_PARALLEL" "$EVICT_NUMBER" "$EVICT_PROMPT" "$EVICT_SEED" "$EVICT_TOKENS" "$EVICT_TOKENS"
                echo "❌ 驱逐请求失败 [$tag], 终止" >&2
                exit 1
            fi

            tag=$(printf "test_%02d_%s_b%s" "$ti" "$name" "$par")
            if run_case "$tag" "$TEST_URL" "$par" "$pnum" "$ptok" "$sseed"; then
                append_csv test "$name" "$ti" "$sseed" "$ptok" "$pnum" "$par" "$TEST_URL" PASS "$RESULT_ROOT/$tag"
                echo "✅ [$ti/$total] $name ${par}并发 PASS"
            else
                append_csv test "$name" "$ti" "$sseed" "$ptok" "$pnum" "$par" "$TEST_URL" FAIL "$RESULT_ROOT/$tag"
                gen_rerun "$tag" "$TEST_URL" "$par" "$pnum" "$ptok" "$sseed"
                echo "❌ [$ti/$total] $name ${par}并发 FAIL" >&2
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        done
    fi
done

#--------------------------summary--------------------------

banner "执行汇总"
echo "汇总 CSV : $CSV_FILE"
echo "参数记录 : $RECORD_FILE"
echo "汇总日志 : $SUMMARY_LOG"
echo "详细日志 : $LOG_DIR/ (每个 case 独立 log)"
if [[ $ONLY_PRESTRESS -eq 1 ]]; then
    echo "仅预埋模式: 预热+预埋已完成, 测试矩阵未执行"
    exit 0
fi
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "全部测试 PASS"
    exit 0
else
    echo "失败 $FAIL_COUNT 个 case, 复现脚本: $RESULT_ROOT/rerun_*.sh"
    exit 1
fi

