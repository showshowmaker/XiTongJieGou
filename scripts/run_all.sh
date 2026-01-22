#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BIN="$BUILD_DIR/matmul_bench"
RESULTS_CSV="$ROOT_DIR/report/results.csv"
PERF_CSV="$ROOT_DIR/report/results_perf.csv"

RECORD=0
for arg in "$@"; do
  case "$arg" in
    --record) RECORD=1 ;;
    *) ;;
  esac
done

mkdir -p "$ROOT_DIR/report"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

cat > "$RESULTS_CSV" <<EOF
case,dtype,kernel,size,time_ms,gflops,repeat,warmup,verify_checked,verify_mismatches,verify_max_abs,verify_max_rel
EOF

"$BIN" --case small --kernel base --verify --csv "$RESULTS_CSV"
"$BIN" --case small --kernel opt --verify --csv "$RESULTS_CSV"
"$BIN" --case large --kernel base --csv "$RESULTS_CSV"
"$BIN" --case large --kernel opt --csv "$RESULTS_CSV"

cat > "$PERF_CSV" <<EOF
case,dtype,kernel,cycles,instructions,branches,branch_misses,cache_references,cache_misses,l1_misses,llc_misses
EOF

bash "$ROOT_DIR/scripts/run_perf_stat.sh" "$BIN" "$PERF_CSV"

if [[ "$RECORD" -eq 1 ]]; then
  bash "$ROOT_DIR/scripts/run_perf_record.sh" "$BIN" small opt
  bash "$ROOT_DIR/scripts/run_perf_record.sh" "$BIN" large opt
fi

echo "Done. Results: $RESULTS_CSV"
echo "Done. Perf stats: $PERF_CSV"
