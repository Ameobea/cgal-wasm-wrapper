# Wasm circumcenter predicate filter

This is the numerical contract for `0004-wasm-circumcenter-filter.patch`. The former `circumcenter_filter.h` experiment is embedded in this patch; it is not a second maintained source file.

The initial browser-worker tests reduced the 40-point alpha-wrap workload by 32–39%, with identical raw output fingerprints. Full composition 62 saw about 13.5% less sampled CGAL alpha-wrap time but only a small whole-program reduction because point generation and simplification dominated. These are workload-specific observations, not universal speed guarantees.

## Why the predicate filter works

Write the plane points as `p,q,r` and the tetrahedron as `a,b,c,d`. Set

```text
u = b-a, v = c-a, w = d-a
D = dot(u, cross(v,w))
N = dot(u,u)*cross(v,w) + dot(v,v)*cross(w,u) + dot(w,w)*cross(u,v)
h = cross(q-p, r-p)
S = dot(h, N + 2*D*(a-p))
```

For a nondegenerate tetrahedron, its circumcenter is `a + N/(2D)`. Consequently,

```text
orientation(p,q,r,circumcenter(a,b,c,d)) = sign(S) * sign(D).
```

`D` and `S` are polynomials of degree three and six. We need only their signs. No division, constructed point, interval endpoint, or square root is required by the fast path. The identity also handles arbitrary plane points, either tetrahedron orientation, and reversed plane orientation; it does not assume the plane is one particular facet.

### Error certificate

The small `AW_bound` scalar stores a floating-point value `v` and a nonnegative expression magnitude `m`. Input magnitude is the exact absolute value of the input coordinate. Addition/subtraction use `m1+m2`; multiplication uses `m1*m2`. All operations use round-to-nearest. This magnitude is **not itself a directed upper bound**; the final certificate accounts for rounding in both computations.

Here is the conservative argument for this particular expression:

1. Expand the expression into signed monomials. Replacing every addition/subtraction with addition of magnitudes produces its unsigned expression magnitude `M`.
2. Each monomial in `D` encounters at most 8 rounding factors; each in `S` at most 20. These counts include rounded coordinate differences and repeated factors. They follow the recurrence `k(add/sub)=max(k1,k2)+1`, `k(mul)=k1+k2+1`. The constant 2 is exact, and counting its multiplication is conservative.
3. With unit roundoff `u0=2^-53`, the absolute forward error is at most `gamma_20*M`, where `gamma_20=20*u0/(1-20*u0)`. The computed positive magnitude satisfies `m >= (1-u0)^20*M`. Thus `gamma_20*m/(1-u0)^20` bounds the value's error.
4. The acceptance factor `2^-40` is more than 400 times that bound. Comparing `abs(v) > 2^-40*m` therefore certifies the sign. The power-of-two multiplication is exact within the guarded exponent range. Both `D` and `S` must pass; exact zero is deliberately left to the fallback.
5. Every input coordinate must be zero or have absolute value in `[2^-100, 2^100]`. Such binary64 inputs are integer multiples of `2^-152`. Through degree six, nonzero intermediates remain multiples of at least `2^-912`, safely above normal underflow. Magnitudes and intermediates are bounded far below overflow (a coarse upper bound is `2^620`). The final error scaling also stays normal. NaNs, infinities and out-of-range coordinates fail the guard. Non-double kernels use the original predicate.

This argument requires strict binary64 arithmetic. Builds disable FP contraction with `-ffp-contract=off`; the patch rejects `__FAST_MATH__` when enabled. Do not enable reassociation or replace these scalar operations with relaxed fused SIMD operations without rechecking the certificate. The patch is active only for Wasm, `CGAL_ALWAYS_ROUND_TO_NEAREST`, and the explicit opt-in macro.

This is a written error analysis with exact-oracle testing, not a machine-checked proof. Changing the polynomial, arithmetic schedule, exponent guard, or number representation requires rechecking it. The fallback preserves CGAL's existing behavior and inherits any defects in the existing FPU patch.

### Predicate checks

`scripts/alpha-wrap-perf/predicate_check.cpp` in the Geoscript checkout compiles to Wasm and compares against the original construction/orientation evaluated with exact `Quotient<MP_Float>` coordinates. Out of 12,000 generated cases, 11,330 have nondegenerate tetrahedra and are checked. There are **zero mismatches**: 2,926 are accepted by the new filter and 8,404 use the original fallback. This deliberately difficult corpus exercises arbitrary planes, facet planes, reversed orientations, nearly flat tetrahedra, exact and one-ULP-near boundary cases, translations, and scales from `2^-140` to `2^140`. Its acceptance percentage is not the workload acceptance rate. Degenerate tetrahedra are excluded because CGAL's original circumcenter predicate also requires nondegeneracy.


Full inputs, raw timings, source analysis and application validation are retained in the Geoscript checkout under `scripts/alpha-wrap-perf/`.
