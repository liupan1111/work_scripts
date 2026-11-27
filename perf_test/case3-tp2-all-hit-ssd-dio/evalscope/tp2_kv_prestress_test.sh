#!/bin/bash
set -o pipefail

#--------------------------default-----------------------------------
# 预热/预埋走 8000, 缓存命中测试走 8000
WARMUP_URL="http://192.168.255.85:8000/v1/chat/completions"
PRESTRESS_URL="http://192.168.255.85:8000/v1/chat/completions"
TEST_URL="http://192.168.255.85:8000/v1/chat/completions"
MODEL="deepseek-14b"
TOKENIZER_PATH="/workspaces/modelscope-yrcache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

MIN_TOKENS=1
MAX_TOKENS=1
REQUEST_MULTIPLIER=1

# 预热: 重启模型后组件加载, 2并发10个8k请求
WARMUP_PARALLEL=2
WARMUP_NUMBER=10
WARMUP_PROMPT=8192
WARMUP_SEED=9831

# 预埋并发 = 测试矩阵中该场景的最大并发 (从 TEST_LIST 自动推导, 加快预埋)

SKIP_WARMUP=0
SKIP_PRESTRESS=0
ONLY_PRESTRESS=0
SKIP_PADDING=0
SCENE_FILTER=""
CASE_FILTER=""

# 场景: 名称:prompt长度:基准请求数:seed (同一场景的预埋与测试共用 seed 和实际请求数)
SCENE_LIST=(
    "4k:4096:64:40"
    "8k:8192:32:41"
    "16k:16384:16:42"
    "32k:32768:8:43"
    "64k:65536:4:44"
    "127k:130048:2:45"
)

# 命中测试矩阵: 场景:并发[:padding] (6个场景按6轮穿插, 共31步: 21个正式用例 + 10个垫场)
# padding 表示垫场, 注释中的请求数为基准值
# 按278880 tokens容量编排, 各场景最大并发为64/32/16/8/4/2
# 前5轮均按4k/8k/16k/32k/64k/127k执行, 补齐项作为垫场, 按对应场景最大并发运行
# 垫场复用同seed/number用于置换缓存, 第6轮只跑4K b64正式用例压轴
TEST_LIST=(
    # 轮1 (1并发, 全场景基线)
    "4k:1"      # 1  轮1 4K   1并发  (64条) 正式
    "8k:1"      # 2  轮1 8K   1并发  (32条) 正式
    "16k:1"     # 3  轮1 16K  1并发  (16条) 正式
    "32k:1"     # 4  轮1 32K  1并发  (8条)  正式
    "64k:1"     # 5  轮1 64K  1并发  (4条)  正式
    "127k:1"    # 6  轮1 127K 1并发  (2条)  正式
    # 轮2 (4并发档, 127k按最大并发垫场)
    "4k:4"      # 7  轮2 4K   4并发  (64条) 正式
    "8k:4"      # 8  轮2 8K   4并发  (32条) 正式
    "16k:4"     # 9  轮2 16K  4并发  (16条) 正式
    "32k:4"     # 10 轮2 32K  4并发  (8条)  正式
    "64k:4"     # 11 轮2 64K  4并发  (4条)  正式
    "127k:2:padding"    # 12 轮2 127K 2并发  (2条)  垫场
    # 轮3 (8并发档, 64k/127k按最大并发垫场)
    "4k:8"      # 13 轮3 4K   8并发  (64条) 正式
    "8k:8"      # 14 轮3 8K   8并发  (32条) 正式
    "16k:8"     # 15 轮3 16K  8并发  (16条) 正式
    "32k:8"     # 16 轮3 32K  8并发  (8条)  正式
    "64k:4:padding"     # 17 轮3 64K  4并发  (4条)  垫场
    "127k:2:padding"    # 18 轮3 127K 2并发  (2条)  垫场
    # 轮4 (16并发档, 32k/64k/127k按最大并发垫场)
    "4k:16"     # 19 轮4 4K   16并发 (64条) 正式
    "8k:16"     # 20 轮4 8K   16并发 (32条) 正式
    "16k:16"    # 21 轮4 16K  16并发 (16条) 正式
    "32k:8:padding"     # 22 轮4 32K  8并发  (8条)  垫场
    "64k:4:padding"     # 23 轮4 64K  4并发  (4条)  垫场
    "127k:2:padding"    # 24 轮4 127K 2并发  (2条)  垫场
    # 轮5 (32并发档, 16k/32k/64k/127k按最大并发垫场)
    "4k:32"     # 25 轮5 4K   32并发 (64条) 正式
    "8k:32"     # 26 轮5 8K   32并发 (32条) 正式
    "16k:16:padding"    # 27 轮5 16K  16并发 (16条) 垫场
    "32k:8:padding"     # 28 轮5 32K  8并发  (8条)  垫场
    "64k:4:padding"     # 29 轮5 64K  4并发  (4条)  垫场
    "127k:2:padding"    # 30 轮5 127K 2并发  (2条)  垫场
    # 轮6 (只跑4K最高档正式用例压轴)
    "4k:64"     # 31 轮6 4K   64并发 (64条) 正式 (最高档压轴)
)

CSV_FILE="./kv_cache_result.csv"

#--------------------------functions-----------------------------------

show_help() {
    cat <<'EOF'
用法: ./kv_prestress_test.sh [选项]

KV cache 预埋 + 缓存命中性能测试
流程:
  1. 模型预热: 2并发 10个 8k 请求 (组件加载)
  2. 最大并发预埋(每场景取测试矩阵中该场景最大并发, 以下为基准请求数): 4k(64条,seed40) / 8k(32条,seed41) / 16k(16条,seed42) / 32k(8条,seed43) / 64k(4条,seed44) / 127k(2条,seed45), 记录首轮业务指标
  3. 缓存命中测试: 同 seed/number, 默认共 31 个 case (21正式 + 10垫场)
     前5轮均按4k/8k/16k/32k/64k/127k顺序执行, 补齐项作为垫场按对应场景最大并发运行
     轮1(1并发): 全场景正式
     -> 轮2(4并发): 4k/8k/16k/32k/64k正式, 127k2并发垫场
     -> 轮3(8并发): 4k/8k/16k/32k正式, 64k4并发 / 127k2并发垫场
     -> 轮4(16并发): 4k/8k/16k正式, 32k8并发 / 64k4并发 / 127k2并发垫场
     -> 轮5(32并发): 4k/8k正式, 16k16并发 / 32k8并发 / 64k4并发 / 127k2并发垫场
     -> 轮6: 仅4k64并发正式用例(最高档压轴)

选项:
  --warmup-url URL       预热地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --prestress-url URL    预埋地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --test-url URL         测试地址, 默认 http://192.168.255.85:8000/v1/chat/completions
  --model NAME           模型名, 默认 deepseek-70b
  --tokenizer-path PATH  tokenizer 路径
  --min-tokens N         输出 token 下限, 默认 1
  --max-tokens N         输出 token 上限, 默认 1
  --request-multiplier N 场景请求数倍数, 默认 1, 用于预埋、正式测试和垫场
  --scene NAME           只预埋和测试指定场景 (4k/8k/16k/32k/64k/127k)
  --case N               只测试原编号 N 的用例 (1-31), 预埋其所属场景
  --skip-padding         跳过垫场用例
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
        --warmup-url)       WARMUP_URL="$2"; shift 2 ;;
        --prestress-url)    PRESTRESS_URL="$2"; shift 2 ;;
        --test-url)         TEST_URL="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --tokenizer-path)   TOKENIZER_PATH="$2"; shift 2 ;;
        --min-tokens)       MIN_TOKENS="$2"; shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2"; shift 2 ;;
        --request-multiplier)
            REQUEST_MULTIPLIER="${2:-}"
            if ! [[ "$REQUEST_MULTIPLIER" =~ ^[1-9][0-9]*$ ]] ||
                [[ "$(( REQUEST_MULTIPLIER ))" != "$REQUEST_MULTIPLIER" ]]; then
                echo "错误: --request-multiplier 需要一个可表示的正整数" >&2
                exit 1
            fi
            shift 2 ;;
        --scene)
            if ! [[ "${2:-}" =~ ^[0-9]+[kK]$ ]]; then
                echo "错误: --scene 需要场景名, 例如 4k" >&2
                exit 1
            fi
            SCENE_FILTER=${2,,}
            shift 2 ;;
        --case)
            case_count=${#TEST_LIST[@]}
            if ! [[ "${2:-}" =~ ^0*([1-9][0-9]*)$ ]] ||
                (( ${#BASH_REMATCH[1]} > ${#case_count} || BASH_REMATCH[1] > case_count )); then
                echo "错误: --case 需要原用例编号 (1-$case_count)" >&2
                exit 1
            fi
            CASE_FILTER=${BASH_REMATCH[1]}
            shift 2 ;;
        --skip-padding)     SKIP_PADDING=1; shift ;;
        --skip-warmup)      SKIP_WARMUP=1; shift ;;
        --skip-prestress)   SKIP_PRESTRESS=1; shift ;;
        --only-prestress)   ONLY_PRESTRESS=1; shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)                  echo "未知选项: $1"; exit 1 ;;
    esac
done

validate_positive_integer "$MIN_TOKENS" "min-tokens" || exit 1
validate_positive_integer "$MAX_TOKENS" "max-tokens" || exit 1

#--------------------------scene--------------------------

if [[ -n "$CASE_FILTER" ]]; then
    IFS=':' read -r case_scene _ case_role <<< "${TEST_LIST[CASE_FILTER - 1]}"
    if [[ -n "$SCENE_FILTER" && "$SCENE_FILTER" != "$case_scene" ]]; then
        echo "错误: 用例 $CASE_FILTER 属于 $case_scene, 与 --scene $SCENE_FILTER 不匹配" >&2
        exit 1
    fi
    if [[ $ONLY_PRESTRESS -eq 0 && $SKIP_PADDING -eq 1 && "$case_role" == padding ]]; then
        echo "错误: 用例 $CASE_FILTER 是垫场, 与 --skip-padding 冲突" >&2
        exit 1
    fi
    SCENE_FILTER=$case_scene
fi

declare -A SCENE_TOKENS SCENE_NUMBER SCENE_SEED SCENE_PARALLEL
ACTIVE_SCENE_LIST=()
SCENE_TAG=""
for entry in "${SCENE_LIST[@]}"; do
    IFS=':' read -r name ptok pnum sseed <<< "$entry"
    SCENE_TOKENS[$name]=$ptok
    scaled_number=$(( pnum * REQUEST_MULTIPLIER ))
    if (( scaled_number / REQUEST_MULTIPLIER != pnum )); then
        echo "错误: --request-multiplier 过大, $name 的请求数超出整数范围" >&2
        exit 1
    fi
    SCENE_NUMBER[$name]=$scaled_number
    SCENE_SEED[$name]=$sseed
    SCENE_TAG="${SCENE_TAG}${name}-"
    if [[ -z "$SCENE_FILTER" || "$name" == "$SCENE_FILTER" ]]; then
        ACTIVE_SCENE_LIST+=("$entry")
    fi
done
SCENE_TAG="${SCENE_TAG%-}"    # 例: 4k-8k-16k-32k-64k-127k
if (( ${#ACTIVE_SCENE_LIST[@]} == 0 )); then
    echo "错误: 未知场景 '$SCENE_FILTER', 请用 --help 查看可选场景" >&2
    exit 1
fi

# 预埋并发取完整测试矩阵的最大值; 执行列表保留原用例编号
ACTIVE_TEST_LIST=()
for i in "${!TEST_LIST[@]}"; do
    IFS=':' read -r name par role <<< "${TEST_LIST[$i]}"
    if [[ -z "${SCENE_PARALLEL[$name]}" ]] || (( par > SCENE_PARALLEL[$name] )); then
        SCENE_PARALLEL[$name]=$par
    fi
    if [[ ( -z "$SCENE_FILTER" || "$name" == "$SCENE_FILTER" ) &&
          ( -z "$CASE_FILTER" || $((i + 1)) == "$CASE_FILTER" ) &&
          ( $SKIP_PADDING -eq 0 || $role != padding ) ]]; then
        ACTIVE_TEST_LIST[$i]=${TEST_LIST[$i]}
    fi
done

command -v evalscope >/dev/null 2>&1 || { echo "错误: 未找到 evalscope 命令" >&2; exit 1; }

#--------------------------prepare dirs & logs--------------------------

# 结果目录命名与测试项关联: 测试项_模型_场景_时间
MODEL_TAG=${MODEL//\//-}
RESULT_ROOT="./results/kv_cache_hit_test_${MODEL_TAG}_${SCENE_TAG}_$(date "+%Y%m%d_%H%M%S")"
LOG_DIR="$RESULT_ROOT/logs"

mkdir -p "$LOG_DIR"
SUMMARY_LOG="$RESULT_ROOT/summary.txt"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

RECORD_FILE="$RESULT_ROOT/prestress_params.txt"
{
    echo "# 预埋参数记录 (测试阶段必须使用相同 seed/number)"
    echo "# model=$MODEL"
    echo "# tokenizer=$TOKENIZER_PATH"
    echo "# min_tokens=$MIN_TOKENS max_tokens=$MAX_TOKENS"
    echo "# request_multiplier=$REQUEST_MULTIPLIER skip_padding=$SKIP_PADDING"
    echo "# warmup: parallel=$WARMUP_PARALLEL number=$WARMUP_NUMBER prompt=$WARMUP_PROMPT seed=$WARMUP_SEED"
    for entry in "${ACTIVE_SCENE_LIST[@]}"; do
        IFS=':' read -r name ptok _ sseed <<< "$entry"
        printf "%-5s prompt_length=%-7s number=%-5s seed=%-4s prestress_parallel=%s\n" "$name" "$ptok" "${SCENE_NUMBER[$name]}" "$sseed" "${SCENE_PARALLEL[$name]}"
    done
} > "$RECORD_FILE"

if [ ! -f "$CSV_FILE" ]; then
    echo "timestamp,model,phase,scene,case,seed,prompt_length,number,parallel,url,TTFT(ms),TPOT(ms),Output Throughput(tok/s),Total Throughput(tok/s),status" > "$CSV_FILE"
fi

#--------------------------run & collect--------------------------

# 运行一次 evalscope 压测: tag url parallel number prompt_len seed
run_case() {
    local tag=$1 url=$2 parallel=$3 number=$4 ptok=$5 seed=$6
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
        --min-tokens "$MIN_TOKENS" \
        --max-tokens "$MAX_TOKENS" \
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

# 失败 case 复现脚本: tag url parallel number ptok seed
gen_rerun() {
    local tag=$1 url=$2 parallel=$3 number=$4 ptok=$5 seed=$6
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
    --min-tokens "$MIN_TOKENS" \\
    --max-tokens "$MAX_TOKENS" \\
    --seed "$seed"
EOF
    chmod +x "$f"
    echo "已生成复现脚本: $f"
}

#--------------------------print plan--------------------------

banner "KV Cache 预埋与命中性能测试"
echo "Model        : $MODEL"
echo "Tokenizer    : $TOKENIZER_PATH"
echo "预热地址     : $WARMUP_URL"
echo "预埋地址     : $PRESTRESS_URL"
echo "测试地址     : $TEST_URL"
echo "输出 tokens  : $MIN_TOKENS ~ $MAX_TOKENS"
echo "请求数倍数   : $REQUEST_MULTIPLIER"
echo "结果目录     : $RESULT_ROOT"
echo "汇总 CSV     : $CSV_FILE"
echo ""
echo "预热: parallel=$WARMUP_PARALLEL number=$WARMUP_NUMBER prompt=$WARMUP_PROMPT seed=$WARMUP_SEED"
echo ""
echo "场景 (预埋与测试共用 seed/number):"
for entry in "${ACTIVE_SCENE_LIST[@]}"; do
    IFS=':' read -r name ptok _ sseed <<< "$entry"
    echo "  $name : prompt=$ptok, number=${SCENE_NUMBER[$name]}, seed=$sseed, 预埋并发=${SCENE_PARALLEL[$name]}"
done
echo ""
echo "测试矩阵 (执行 ${#ACTIVE_TEST_LIST[@]} 个用例):"
for i in "${!ACTIVE_TEST_LIST[@]}"; do
    IFS=':' read -r name par _ <<< "${ACTIVE_TEST_LIST[$i]}"
    echo "  $((i + 1)). $name ${par}并发 (number=${SCENE_NUMBER[$name]}, seed=${SCENE_SEED[$name]})"
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

#--------------------------2. prestress--------------------------

if [[ $SKIP_PRESTRESS -eq 0 ]]; then
    banner "阶段2 最大并发数据预埋 (记录首次业务请求性能)"
    for entry in "${ACTIVE_SCENE_LIST[@]}"; do
        IFS=':' read -r name ptok _ sseed <<< "$entry"
        pnum=${SCENE_NUMBER[$name]}
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
    done
else
    echo ""
    echo "跳过预埋阶段 (--skip-prestress)"
fi

#--------------------------3. cache hit test--------------------------

if [[ $ONLY_PRESTRESS -eq 0 ]]; then
    banner "阶段3 缓存命中性能测试"
    ti=0
    total=${#ACTIVE_TEST_LIST[@]}
    for i in "${!ACTIVE_TEST_LIST[@]}"; do
        ti=$((ti + 1))
        case_number=$((i + 1))
        IFS=':' read -r name par _ <<< "${ACTIVE_TEST_LIST[$i]}"
        ptok=${SCENE_TOKENS[$name]}
        pnum=${SCENE_NUMBER[$name]}
        sseed=${SCENE_SEED[$name]}
        tag=$(printf "test_%02d_%s_b%s" "$case_number" "$name" "$par")
        if run_case "$tag" "$TEST_URL" "$par" "$pnum" "$ptok" "$sseed"; then
            append_csv test "$name" "$case_number" "$sseed" "$ptok" "$pnum" "$par" "$TEST_URL" PASS "$RESULT_ROOT/$tag"
            echo "✅ [$ti/$total] $name ${par}并发 PASS"
        else
            append_csv test "$name" "$case_number" "$sseed" "$ptok" "$pnum" "$par" "$TEST_URL" FAIL "$RESULT_ROOT/$tag"
            gen_rerun "$tag" "$TEST_URL" "$par" "$pnum" "$ptok" "$sseed"
            echo "❌ [$ti/$total] $name ${par}并发 FAIL" >&2
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
fi

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

