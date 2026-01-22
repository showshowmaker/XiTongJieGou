#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./build/matmul_bench}
OUT=${2:-report/results_perf.csv}

EVENTS="cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses"

mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$OUT" ]]; then
  echo "case,dtype,kernel,cycles,instructions,branches,branch_misses,cache_references,cache_misses,l1_misses,llc_misses" > "$OUT"
fi

run_case() {
  local case_label=$1
  local dtype=$2
  local kernel=$3
  shift 3

  local tmp
  tmp=$(mktemp)
  perf stat -x , -e "$EVENTS" -- "$BIN" "$@" 2> "$tmp" 1> /dev/null

  get_event() {
    local ev=$1
    awk -F, -v ev="$ev" '$3==ev {gsub(/ /, "", $1); print $1; exit}' "$tmp"
  }

  local cycles instructions branches branch_misses cache_refs cache_misses l1_misses llc_misses
  cycles=$(get_event cycles)
  instructions=$(get_event instructions)
  branches=$(get_event branches)
  branch_misses=$(get_event branch-misses)
  cache_refs=$(get_event cache-references)
  cache_misses=$(get_event cache-misses)
  l1_misses=$(get_event L1-dcache-load-misses)
  llc_misses=$(get_event LLC-load-misses)

  echo "$case_label,$dtype,$kernel,$cycles,$instructions,$branches,$branch_misses,$cache_refs,$cache_misses,$l1_misses,$llc_misses" >> "$OUT"
  rm -f "$tmp"
}

run_case small32 fp32 base --case small --kernel base
run_case small32 fp32 opt --case small --kernel opt
run_case large4096 fp64 base --case large --kernel base
run_case large4096 fp64 opt --case large --kernel opt
