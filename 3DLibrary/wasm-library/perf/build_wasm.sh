#!/usr/bin/env bash
# Build the alpha_wrap repro to Wasm with the SAME flags as the real app build
# (CMakeLists.txt), so Node/V8 timing is a faithful proxy for the Chrome worker.
# No GMP includes -> CGAL uses Quotient<MP_Float> (header-only), matching the real index.wasm.
# Runs wasm-opt -O4 like the real Justfile so codegen matches.
#   ./build_wasm.sh baseline
#   ./build_wasm.sh nobarriers -DCGAL_IA_RTN_NO_BARRIERS
set -euo pipefail
cd "$(dirname "$0")"
LIB=../libraries
SUFFIX="${1:-baseline}"
shift || true
source ~/emsdk/emsdk_env.sh >/dev/null 2>&1

emcc -std=c++14 -O3 -DNDEBUG -w \
  -msimd128 -mrelaxed-simd -msse4.2 -fwasm-exceptions \
  -DCGAL_ALWAYS_ROUND_TO_NEAREST \
  "$@" \
  -I"$LIB/CGAL-5.6.3/include" \
  -I"$LIB/boost_1_72_0" \
  -I"$LIB/eigen-3.3.9" \
  -s ALLOW_MEMORY_GROWTH=1 -s NO_EXIT_RUNTIME=0 -s EXIT_RUNTIME=1 \
  -s ENVIRONMENT=node -s TOTAL_STACK=33554432 \
  native_bench.cpp -o "nb_$SUFFIX.js"

# Match the real pipeline's post-link wasm-opt -O4.
wasm-opt "nb_$SUFFIX.wasm" -O4 --enable-exception-handling --enable-bulk-memory \
  --enable-simd --enable-relaxed-simd --enable-nontrapping-float-to-int \
  -o "nb_$SUFFIX.wasm" 2>/dev/null || echo "(wasm-opt skipped/failed)"
echo "built nb_$SUFFIX.{js,wasm}"
