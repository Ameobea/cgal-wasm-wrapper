cmake_minimum_required(VERSION 3.5...3.31)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cgalTest)

add_compile_definitions(CGAL_ALWAYS_ROUND_TO_NEAREST)
# Drop the interval-arithmetic opacify barriers: in ROUND_TO_NEAREST mode they guard against
# a hardware directed-rounding switch that never happens, so they are correctness-neutral
# (verified byte-identical across alpha_wrap / CDT / boolean) but on Wasm force a linear-memory
# spill+reload per interval op. ~10% faster alpha_wrap. See perf/FINDINGS.md, perf/EXPERIMENTS.md.
add_compile_definitions(CGAL_IA_RTN_NO_BARRIERS)
# See patches/FINDINGS.md for the error certificate and Wasm measurements.
option(CGAL_WASM_CIRCUMCENTER_FILTER "Enable the certified alpha-wrap predicate fast path" ON)
if(CGAL_WASM_CIRCUMCENTER_FILTER)
  add_compile_definitions(CGAL_AW3_WASM_CIRCUMCENTER_FILTER)
endif()
# No GMP, and no Boost.Multiprecision either: exact fallbacks then use MP_Float/Quotient<MP_Float>
# as the 5.6 build did. Boost's cpp_rational is ~2x slower in wasm (gcd normalization + soft-float
# long double conversions) on field-type predicates, though its cpp_float makes delaunay_remesh
# ~25% faster; it also adds ~0.7MB of wasm. Eigen's x86 SSE path uses inline asm wasm can't take,
# and the only Eigen use here is tiny (region-growing PCA).
add_compile_definitions(CGAL_DISABLE_GMP=1 CGAL_DO_NOT_USE_BOOST_MP EIGEN_DONT_VECTORIZE)

# Header-only deps; versions must match install_libraries.sh and the Justfile's patch-deps.
set(LIBS ${CMAKE_CURRENT_SOURCE_DIR}/libraries)
include_directories(${LIBS}/CGAL-6.2/include ${LIBS}/boost_1_89_0 ${LIBS}/eigen-3.4.0)

add_executable(main main.cpp)
add_definitions(-w)

set_target_properties(main
PROPERTIES OUTPUT_NAME "index"
# -fwasm-exceptions -sASSERTIONS -sNO_DISABLE_EXCEPTION_CATCHING
LINK_FLAGS "-fwasm-exceptions -O3 -msimd128 -mrelaxed-simd --llvm-features=+simd128 -msse4.2 -s ALLOW_MEMORY_GROWTH=1 -profiling -flto --bind -s EXPORT_NAME=\"CGAL\" -s MODULARIZE=1 -s NO_EXIT_RUNTIME=1 -s NO_FILESYSTEM=1"
)
target_compile_options(main PRIVATE -ffp-contract=off -fwasm-exceptions -O3 -msimd128 -mrelaxed-simd -llvm-features=+simd128 -msse4.2)
