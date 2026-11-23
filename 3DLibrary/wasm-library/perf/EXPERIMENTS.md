# alpha_wrap perf — experiment log

Numbers are **best-of-N wall time** of `CGAL::alpha_wrap_3` only (point gen excluded).
Two measurement rigs:
- **native** (`build_native.sh`): clang++ -O3 on x86. Mirrors the algorithm but NOT Wasm
  codegen — used for **correctness/parity** (byte-exact output fingerprints) and host
  profiling. Note: on x86 `IA_opacify` is a cheap *register* barrier, so barrier-removal
  shows ~0 here by design.
- **wasm/node** (`build_wasm.sh` → `node nb_*.js`): emcc with the **same flags as the app**
  (`-O3 -msimd128 -mrelaxed-simd -msse4.2 -fwasm-exceptions`) + `wasm-opt -O4`. Node and
  Chrome share V8, so this is a faithful proxy. Baseline torus/1550 = 824 ms in Node vs
  884 ms in the captured Chrome worker trace → good match.

Deterministic point clouds (fixed LCG seed) so output is diffable across variants:
`torus_shell`, `sphere_shell`, `noise_shell` in `native_bench.cpp`.

---

## Baseline (no changes)

| case | native ms | wasm/node ms | verts / faces |
|---|---:|---:|---|
| torus_shell 1550, α=1/100, off=1/100 | 548 | 824 | 7303 / 14266 |
| sphere_shell 8000, α=1/140, off=1/250 | 2135 | — | 24843 / 45134 |

Output fingerprint is stable across reps on each platform (deterministic). Native vs Wasm
fingerprints differ in coordinate last-bits (different FP codegen / SIMD / contraction) but
**vert/face counts are identical** → the exact-predicate-driven combinatorics are identical,
which is the robustness invariant that matters.

---

## H1 — drop IA opacify barriers in CGAL_ALWAYS_ROUND_TO_NEAREST mode  ✅ WIN

**Change** (`cgal-fpu_PATCHED.h`, opt-in `CGAL_IA_RTN_NO_BARRIERS`): in round-to-nearest
mode make `CGAL_IA_STOP_CPROP(x)` and `CGAL_IA_FORCE_TO_DOUBLE(x)` identity instead of
`IA_opacify(x)`.

**Rationale:** the barriers exist to stop the compiler folding/migrating interval ops
across a *hardware directed-rounding switch*. In `CGAL_ALWAYS_ROUND_TO_NEAREST` there is no
such switch — compile-time and runtime rounding are both round-to-nearest, so folding is
value-identical, and the directed widening that makes intervals valid is applied explicitly
and **asymmetrically** by `nextUp()` (a bit-cast that is itself an optimization barrier).
The CGAL source comment confirms the barriers guard the `-fno-rounding-math` / round-up
case. On Wasm the patched `IA_opacify` is the `asm volatile("":"+m"(x))` form, forcing a
linear-memory spill+reload **per interval op**; that's what we remove.

**Correctness:** native output **byte-for-byte identical** (fingerprint unchanged) on
torus_shell and sphere_shell; Wasm output **byte-for-byte identical** on torus_shell. No
combinatorial or coordinate change.

**Perf:**

| case | rig | baseline | nobarriers | speedup |
|---|---|---:|---:|---:|
| torus_shell 1550 α=1/100 | native | 533 | 526 | ~1% (expected ~0 on x86) |
| sphere_shell 8000 α=1/140 | native | 2135 | 2144 | ~0% (expected) |
| torus_shell 1550 α=1/100 | **wasm/node** | **829** | **749** | **9.6%** |

Multi-scale Wasm/Node sweep (reps=3, best-of), **parity OK (byte-identical) at every scale**:

| case | baseline ms | nobarriers ms | speedup |
|---|---:|---:|---:|
| torus_shell 1550 α=1/100 | 837 | 742 | 11.4% |
| torus_shell 4000 α=1/120 | 1989 | 1785 | 10.3% |
| sphere_shell 8000 α=1/140 | 3294 | 2966 | 10.0% |
| noise_shell 12000 α=1/160 | 6056 | 5466 | 9.7% |

Consistent **~10%**, tapering slightly at larger scale (interval-arith share drops as
static-filter/AABB work grows — matches the cost-balance prediction).

**Status:** validated win, safe for alpha_wrap. Before making it the default under
`CGAL_ALWAYS_ROUND_TO_NEAREST` (affects ALL interval arithmetic), parity-checking the other
kernels/builtins (CDT EPICK, boolean EPECK/Lazy) — see kernel_parity below.

---

## Shipped: H1 enabled in the app build

- `CMakeLists.txt`: `add_compile_definitions(CGAL_IA_RTN_NO_BARRIERS)`.
- Full `just build` succeeds with the flag (`flags.make` shows it). Real artifact
  (`build/index.{js,wasm}`) smoke-tested in Node via embind: alpha_wrap on 1550-pt torus
  shell → 849 ms, valid closed mesh (7268v/14352f), SMOKE OK.
- Installed to `~/dream/src/viz/wasm/cgal/`. **Only `index.wasm` changed** — `index.js`
  glue is byte-identical (pure codegen change, drop-in). Revert with
  `git checkout src/viz/wasm/cgal/index.wasm`.
- Parity across kernels (`kernel_parity.cpp`, baseline vs nobarriers, byte-identical):
  CDT+refine EPICK `094060ea32611e02`; polygon-boolean EPECK/Lazy `5b215c23a78d282a`.

## H2b — Wasm-SIMD Interval_nt ({-inf,sup} in one v128 + single SIMD nextUp)  ❌ NOT WORTH IT

`microbench_interval.cpp` (standalone, wasm/node): scalar post-H1 vs SIMD interval
multiply+add chain, 20M ops. **Parity bit-exact (0 mismatches)** — confirms the SIMD
formulation (brutal all-products multiply + one `simd_nextUp`, `add` = one f64x2 add + one
`simd_nextUp`) is byte-identical to CGAL's scalar+nextUp. But **speedup only 1.05×**.

Why marginal: the scalar multiply's sign-case branches compute only 1–2 products for
definite-sign operands; the SIMD "brutal" path always computes 4 products + maxes (more
work, just parallel), and the serial dependency exposes SIMD op latency. An SSE4-style
blendv 2-product SIMD multiply might reach ~1.1–1.2×, but interval primitives are only ~27%
of post-H1 runtime, so even an optimistic 1.3× nets ~2–3% total — below the bar for an
invasive, foundational rewrite of `Interval_nt.h`. **Rejected.**

## H3/H4 — robust circumcenter / static-filter Do_intersect_3 / AABB (non-rounding ~66%)

These dominate the remaining runtime and are CGAL-internal alpha_wrap algorithm costs
(AABB-tree oracle queries, point/sphere `Do_intersect_3`, robust circumcenter
construction), NOT rounding-related. They grow their share at larger point counts (why H1's
% tapers). Optimizing them means changing alpha_wrap's algorithm / CGAL internals — high
effort, high risk, orthogonal to the rounding-mode question. Not pursued; documented as the
real ceiling on further alpha_wrap speedup.

## Voxel-downsample input before wrapping — CONDITIONAL win (algorithmic, no code shipped)

`alpha_wrap` cannot resolve detail finer than `alpha`, so input points spaced ≪ alpha are
redundant. Voxel-downsampling to ~alpha spacing (`AW_DOWNSAMPLE=<frac>` in `native_bench`)
helps **only when the input is denser than the alpha resolution**:

| input | full | ds@1.0·alpha | output change | speedup |
|---|---:|---|---|---:|
| dense (torus tube 40k pts, α=1/60 → 2517 out-verts; input 16× output) | 227 ms | 40k→8478 pts, 170 ms | 2517→2551 verts (~1%) | **25%** |
| thin shell (noise 60k pts, α=1/200 → 195762 out-verts; output 3× input) | 20914 ms | 60k→59732 pts | ~0 | **~0%** |

Takeaway: cost is driven by **output complexity** (Steiner points ∝ surface_area / alpha²)
plus input handling. Downsampling only removes input redundancy. For thin-shell / noise-band
clouds (the user's heavy cases — output ≥ input) it does nothing; for over-sampled
volumetric clouds it's a large free win. Heuristic for the user: if input point count ≫ wrap
output vertex count, downsample to ~alpha spacing first. Not shipped as code (situational +
a quality/UX choice); could be a Geoscript `voxel_downsample` builtin or an auto-pre-pass.

## EPECK pathBoolean2D (Quotient<MP_Float>) — note for future

Boolean ops use exact constructions via `Quotient<MP_Float>` (no GMP in the Wasm build).
H1 already speeds up their interval-filter tier. The exact-construction tier uses MP_Float
(header-only, slower than GMP). Compiling GMP to Wasm and switching EPECK to `Gmpq`/`Mpzf`
would speed up heavy boolean workloads, but is a build-infrastructure project and only helps
the boolean builtin. Not pursued.
