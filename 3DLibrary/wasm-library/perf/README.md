# alpha_wrap (CGAL/Wasm) performance investigation

Goal: make Geoscript's CGAL-backed builtins (esp. `alpha_wrap`) faster in Wasm, and
understand the directed-rounding / exact-kernel "tax" that the no-FPU-rounding-mode Wasm
target pays.

**Read `FINDINGS.md` first** (analysis + the OUTCOME summary), then `EXPERIMENTS.md`
(before/after numbers and every hypothesis tried, including the rejected ones).

## Result

- **~10% faster `alpha_wrap`** (and CDT / remesh / boolean — anything interval-filtered),
  byte-identical output. Shipped: `CGAL_IA_RTN_NO_BARRIERS` in `../CMakeLists.txt` drops the
  interval-arithmetic optimization barriers, which are correctness-neutral in
  `CGAL_ALWAYS_ROUND_TO_NEAREST` mode but on Wasm force a linear-memory spill+reload per
  interval op. Rebuilt `index.wasm` installed to `~/dream/src/viz/wasm/cgal/`.
- The rounding tax is ~34% of runtime (biggest single bucket) but not the majority; static
  filters / AABB oracle / data structures are the rest. ~10% is near the ceiling for the
  rounding angle. A Wasm-SIMD interval type was prototyped and rejected (~1.05× on
  primitives). See FINDINGS/EXPERIMENTS.

## Files

| file | what |
|---|---|
| `FINDINGS.md` | mechanism, why exact predicates are required, profile breakdown, outcome |
| `EXPERIMENTS.md` | benchmark log: baseline, H1 (win), H2b SIMD (rejected), downsample, EPECK note |
| `native_bench.cpp` / `build_native.sh` | host alpha_wrap repro — fast iteration + **output parity** (byte-exact fingerprints). `AW_DOWNSAMPLE=<frac>` voxel-downsamples to frac*alpha. |
| `build_wasm.sh` / `run_ab.mjs` | emcc build (app flags + wasm-opt -O4) → Node/V8 A/B. Node==Chrome's engine, so faithful. |
| `kernel_parity.cpp` | parity harness for CDT (EPICK) + polygon-boolean (EPECK/Lazy) |
| `microbench_interval.cpp` | standalone scalar-vs-SIMD interval multiply/add (the H2b go/no-go) |
| `parse*.mjs` | Chrome trace → per-thread self-time + category breakdown (used on the user's capture) |
| `smoke_real.cjs` | loads the REAL built `index.wasm` via embind in Node and runs alpha_wrap |

## Reproduce

```bash
# native parity + perf (host)
./build_native.sh nobarriers -DCGAL_IA_RTN_NO_BARRIERS
./native_bench_nobarriers torus_shell 1550 100 100 6        # case npoints alphaInv offInv reps

# wasm A/B in Node (faithful to Chrome)
source ~/emsdk/emsdk_env.sh
./build_wasm.sh baseline
./build_wasm.sh nobarriers -DCGAL_IA_RTN_NO_BARRIERS
node run_ab.mjs 3 baseline nobarriers

# the H2b SIMD-interval microbench
emcc -O3 -msimd128 -msse4.2 -DNDEBUG microbench_interval.cpp -o mbi.js && node mbi.js
```

The build flag is the only change shipped; `cgal-fpu_PATCHED.h` adds the opt-in
`CGAL_IA_RTN_NO_BARRIERS` guard (no behavior change unless defined).
