#!/usr/bin/env bash
set -euo pipefail

BIN=${1:-./build/matmul_bench}
CASE=${2:-small}
KERNEL=${3:-opt}

perf record -g -- "$BIN" --case "$CASE" --kernel "$KERNEL"

echo "Run 'perf report' or 'perf annotate' to inspect hotspots." 
