#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./build/matmul_bench}
CSV=${2:-report/results.csv}

"$BIN" --case small --kernel base --verify --csv "$CSV"
"$BIN" --case small --kernel opt --verify --csv "$CSV"
"$BIN" --case large --kernel base --csv "$CSV"
"$BIN" --case large --kernel opt --csv "$CSV"
