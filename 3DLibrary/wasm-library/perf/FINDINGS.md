# CGAL/Wasm alpha_wrap perf investigation — FINDINGS

Living doc. Ground-truth numbers come from a real browser Wasm run; see `EXPERIMENTS.md`
for the before/after benchmark log.

## OUTCOME (read this first)

- **Shipped a validated ~10% speedup** for `alpha_wrap` (and all interval-filtered CGAL
  builtins: CDT, remeshing, boolean ops) by dropping the interval-arithmetic optimization
  barriers in `CGAL_ALWAYS_ROUND_TO_NEAREST` mode (H1). They guard a hardware
  directed-rounding switch that never happens on Wasm, so they're correctness-neutral
  (proven **byte-identical** output across alpha_wrap/CDT/boolean, native+Wasm, 4 geometries,
  multiple scales) but on Wasm forced a linear-memory spill+reload per interval op.
  Enabled via `CGAL_IA_RTN_NO_BARRIERS` in `CMakeLists.txt`; rebuilt+installed `index.wasm`.
- **The rounding tax is NOT the majority of runtime.** It's the biggest single category
  (~34% pre-H1), but static filters (~26%, pure double, no rounding), AABB-tree oracle
  queries (~14%), and alpha-wrap data structures (~12%) together dominate. So ~10% is close
  to the realistic ceiling for the rounding-mode angle; there is **no 2× hiding in the
  rounding path**.
- **Kernel choice is already optimal.** alpha_wrap uses EPICK (exact predicates, inexact
  constructions) — the minimal correct kernel. Exact predicates are *required* for Delaunay
  robustness, and interval arithmetic *is* their fast path; no kernel avoids it while
  staying robust. Directed rounding is integral to interval arithmetic (and `nextUp` is
  already near-minimal per op), but is NOT integral to exact predicates in general (static
  filters need none) — the residual interval cost here is mostly robust-circumcenter
  *constructions*, which are inherently interval-based.
- **A Wasm-SIMD interval type (H2b) was prototyped and rejected** — bit-exact but only
  ~1.05× on the primitives → ~1–2% total. Not worth an invasive `Interval_nt.h` rewrite.
- **Bigger speedups would require attacking the non-rounding ~66%** (alpha_wrap's AABB
  oracle / static-filter intersection / circumcenter), i.e. CGAL-internal algorithm work,
  or reducing the workload (fewer Steiner points = coarser α, a quality tradeoff). The
  rounding-mode lever is largely spent at ~10%.


## TL;DR (state as of initial analysis, before any change)

- `alpha_wrap` in Geotoy uses **`Exact_predicates_inexact_constructions_kernel` (EPICK)**,
  *not* EPECK. (main.cpp:72). EPECK is used **only** for the 2D polygon boolean ops
  (`pathBoolean2D`). So alpha_wrap's cost is **predicate filtering + robust constructions
  via interval arithmetic**, *not* multiprecision (GMP/boost) exact constructions.
- The "rounding-mode tax" (CGAL's `CGAL_ALWAYS_ROUND_TO_NEAREST` + `nextUp`/`nextafter`
  directed-rounding emulation) is real and is the **single biggest cost category at ~34%**
  of active compute — but it is **not** the whole story. Static filters (pure double, no
  rounding) are ~26%, AABB-tree queries ~14%, alpha-wrap data structures ~12%.
- Realistic ceiling for "eliminate the directed-rounding tax entirely" is therefore on the
  order of **~15–20%**, not 2×. A larger speedup needs to also attack the static-filter /
  AABB / algorithmic side, or reduce the *number* of operations (Steiner points).

## Why exact predicates are required at all (the robustness problem)

Delaunay-based algorithms (the 3D Delaunay triangulation underneath `alpha_wrap_3`, the 2D
CDT, etc.) are **combinatorial algorithms driven by the signs of geometric predicates**
(`orientation_3` = sign of a 4×4 determinant; `side_of_oriented_sphere_3` = sign of a 5×5
determinant; etc.). If a predicate returns the wrong sign because of floating-point error,
the algorithm can take globally inconsistent branches → crash, infinite loop, or corrupt
mesh. This is the classic *robustness problem* in computational geometry, and it is why a
naive `Simple_cartesian<double>` kernel "collapses or infinite-loops" (as the user
observed). The fix is **exact predicates**: guarantee the *sign* is always correct.

EPICK gives **exact predicates** (robust combinatorics) with **inexact constructions**
(coordinates of constructed points are plain `double`). That is exactly what `alpha_wrap_3`
wants, and is what CGAL's own alpha-wrap examples use. So **the kernel choice is already
correct/minimal** — there is no kernel that keeps robustness while avoiding the interval
machinery, because *interval arithmetic IS the fast path of exact predicates*.

## How CGAL makes "exact" predicates fast — and where rounding mode enters

A filtered predicate is evaluated in up to three tiers, cheapest first:

1. **Static / semi-static filters** (`Static_filters_predicates::*`): plain `double`
   arithmetic with a precomputed/running error bound. **No rounding-mode control needed**
   (round-to-nearest only). Resolves the vast majority of "easy" (non-degenerate) cases.
2. **Interval arithmetic filter** (`CGAL::Interval_nt`): recompute with an enclosing
   interval `[lo,hi]`; if the interval doesn't straddle 0 the sign is certain. **This is
   the tier that needs directed rounding** — `hi` must round toward +∞ and `lo` toward −∞
   so the interval is a guaranteed enclosure.
3. **Exact** (multiprecision): last resort, only when the interval still straddles 0.

On normal CPUs tier 2 uses the hardware FPU rounding-mode register (`fesetround`).
**WebAssembly has no rounding-mode control**, so CGAL compiles with
`CGAL_ALWAYS_ROUND_TO_NEAREST`: it keeps the FPU in round-to-nearest and, after every
interval `+ − × ÷ √`, nudges the result outward by one ULP with `nextafter(x,+∞)` (patched
here to a hand-rolled branchless `nextUp`). That per-op nudge (+ the optimization barriers
around it) is the "tax".

### Is the rounding mode "integral" or an "impl detail"?

Both, precisely:
- It is **integral to interval arithmetic** (tier 2). You cannot do correct interval
  arithmetic without directed rounding *of some form*; `nextUp` is the round-to-nearest
  emulation and is already near-minimal per op (1 ULP is the tightest correct widening).
- It is **not integral to exact predicates as a concept** — tier 1 (static filters) proves
  CGAL has a fully round-to-nearest filtering path. The reason we still pay tier 2 a lot
  here is mostly **constructions**, not predicate filtering: alpha_wrap computes **robust
  circumcenters** (`Robust_circumcenter_filtered_traits`, `circumcenterC3<Interval_nt>`,
  `Construct_vector_3<Simple_cartesian<Interval_nt>>`) for its tetrahedra, and those are
  inherently interval-based constructions (they produce *values*, which static filters,
  being sign-only, cannot replace).

## Ground-truth profile (real browser Wasm)

Source: user-captured Chrome trace `~/Downloads/alpha-wrap.json.gz` of the ~900 ms Geoscript
program (1550 points on a torus shell, `alpha_wrap(alpha=1/100, offset=1/100)`), Geoscript
worker isolate (pid 135586, tid 21). `alphaWrapPointCloud` = **884 ms inclusive of the
934 ms total eval (95%)** — alpha_wrap fully dominates; Geoscript point gen/filter is ~4.5%.

Self-time by category (% of 963 ms active compute), see `perf/parse3.mjs`:

| Category | self ms | % active | Rounding tax? |
|---|---:|---:|---|
| **Interval arithmetic** (operator\*, determinant, circumcenterC3, square, orientationC3, /, vector) | 326 | **33.8%** | **yes — nextUp lives here** |
| **Static filters** (Do_intersect_3, Side_of_oriented_sphere_3, Compare_squared_radius_3) | 250 | 25.9% | no (pure double) |
| AABB tree traversal / Construct_bbox_3(Sphere) | 133 | 13.8% | no |
| Alpha_wrap core / pairing_heap / triangulation DS | 114 | 11.8% | no |
| Other (uncategorized) | 73 | 7.6% | — |
| Geoscript point gen/filter/eval | 43 | 4.5% | no |
| Alloc (malloc/free) | 13 | 1.3% | no |
| Filtered_predicate / Uncertain machinery | 12 | 1.3% | partial |

Single hottest leaf: **`Interval_nt<true>::operator*` = 133 ms (13.8%)** — and per the user,
`nextafter` now *inlines* into these interval ops (since the optimization commits), so its
cost is smeared across operator\*/square/determinant/circumcenter rather than showing as its
own frame. So a large fraction of that 133 ms is the directed-rounding nudge + barriers.

## The FPU patch vs upstream CGAL (diff confirmed)

`cgal-fpu_PATCHED.h` vs pristine CGAL v5.6.3 `Number_types/include/CGAL/FPU.h` differs in
exactly three things — the `CGAL_ALWAYS_ROUND_TO_NEAREST` + `nextafter` path itself is
**shipped upstream** as CGAL's baseline for platforms without rounding-mode control:

1. `#ifdef CGAL_HAS_SSE2` → `#if defined(CGAL_HAS_SSE2) && !defined(__wasm__)` in
   `IA_opacify` / `IA_force_to_double` (so it stops emitting x86 SSE inline-asm barriers on
   Wasm and falls through to the `"+m"` memory-barrier form).
2. `IA_up()` in round-to-nearest mode: `return nextafter(d, +inf)` → `return nextUp(d)`
   (hand-rolled branchless successor; commit a50bf91). This is the perf win that made
   `nextafter` inline.
3. `-0` handling fix in `nextUp` (commit 1499e54).

## Optimization hypotheses (to test, ranked by promise/effort)

- **H1 — Drop the optimization barriers in ROUND_TO_NEAREST mode.** In
  `CGAL_ALWAYS_ROUND_TO_NEAREST` the compile-time and runtime rounding modes are *both*
  round-to-nearest, so the `IA_opacify` / `CGAL_IA_STOP_CPROP` / `IA_force_to_double`
  barriers (which exist to stop the compiler from constant-folding or migrating ops across a
  `fesetround` that *doesn't happen here*) appear to be **unnecessary for correctness**, and
  on Wasm the `asm volatile("" : "+m"(x))` form forces a local→linear-memory spill+reload on
  every interval op. `nextUp` itself is a bit-cast and acts as an optimization barrier, and
  in round-to-nearest `fl((−a)−b) == fl(−(a+b))`, so the transforms the barriers guard
  against are sign-symmetric and safe. Hypothesis: removing/relaxing these barriers (guarded
  by `CGAL_ALWAYS_ROUND_TO_NEAREST`) speeds up the 34% interval-arithmetic bucket
  meaningfully with no correctness loss. **Must verify byte-for-byte output parity.**
- **H2 — Cheaper/fewer roundings in `Interval_nt::operator*` (the 14% hot leaf).** The
  standard interval multiply does several products + nudges; check whether the Wasm codegen
  is doing redundant work (e.g. the sign-select branches, or 2× nextUp where 1 fused widen
  would do).
- **H3 — Robust circumcenter cost.** `Robust_circumcenter_filtered_traits` drives a large
  share of the interval ops. Investigate whether a cheaper construction (or caching) is
  possible for the common non-degenerate case.
- **H4 — Static-filter `Do_intersect_3` (26%).** Biggest non-rounding cost. Algorithmic /
  AABB-side; understand what it is doing in the alpha-wrap inner loop and whether it can be
  cut. Independent of the rounding question but matters for the headline number.
- **H5 — Build flags.** Confirm `-ffast-math` is NOT enabled (it would break interval
  correctness AND nextUp's bit tricks). Check `-msimd128`/`-mrelaxed-simd` effects on the
  interval code and whether wasm SIMD can vectorize the two interval bounds.

## Other CGAL-backed Geoscript builtins that hit this path

From `src/viz/wasm/geoscript` + `main.cpp` bindings. EPICK (interval-filter tax, same as
alpha_wrap): `alpha_wrap`, 2D CDT (`triangulatePolygon2D[WithHoles]`, used by tessellate /
path triangulation), `isotropic_remesh`, `delaunay_remesh`, `remesh_planar_patches`,
subdivision smooths. EPECK (exact *constructions*, heavier — multiprecision via boost cpp_int
since GMP is not linked in the Wasm build): `pathBoolean2D` (2D polygon boolean ops). The
EPECK path is a separate, likely-worse offender per-op and worth a look after alpha_wrap.
