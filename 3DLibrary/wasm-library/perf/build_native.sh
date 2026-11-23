#!/usr/bin/env bash
# Build the native alpha_wrap repro against the (already-patched) bundled CGAL headers.
# Mirrors the Wasm config: EPICK, CGAL_ALWAYS_ROUND_TO_NEAREST, no GMP, -O3 -DNDEBUG.
# Pass an output suffix + extra defines to build variants, e.g.:
#   ./build_native.sh baseline
#   ./build_native.sh nobarriers -DCGAL_IA_NO_OPACIFY_IN_RTN
set -euo pipefail
cd "$(dirname "$0")"
LIB=../libraries
SUFFIX="${1:-baseline}"
shift || true
CXX="${CXX:-clang++}"
"$CXX" -std=c++14 -O3 -DNDEBUG -w \
  -DCGAL_ALWAYS_ROUND_TO_NEAREST \
  "$@" \
  -I"$LIB/CGAL-5.6.3/include" \
  -I"$LIB/boost_1_72_0" \
  -I"$LIB/eigen-3.3.9" \
  native_bench.cpp -lgmp -lmpfr -o "native_bench_$SUFFIX"
echo "built native_bench_$SUFFIX"
