set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="/test/venv/bin:$PATH"

runner=./tp2_kv_prestress_test.sh
bash "$runner" --only-prestress --scene 32k
bash "$runner" --only-prestress --skip-warmup --scene 64k
bash "$runner" --only-prestress --skip-warmup --scene 127k

sleep 90

bash "$runner" --case 16 --skip-prestress --skip-warmup
bash "$runner" --case 11 --skip-prestress --skip-warmup
bash "$runner" --case 12 --skip-prestress --skip-warmup

