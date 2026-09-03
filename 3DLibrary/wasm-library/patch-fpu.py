#!/usr/bin/env python3
"""Patches CGAL's FPU.h for Wasm: no SSE2 rounding-mode intrinsics, a bit-twiddling nextUp()
instead of nextafter(), and CGAL_IA_RTN_NO_BARRIERS to drop the interval-arithmetic opacify
barriers that are correctness-neutral under CGAL_ALWAYS_ROUND_TO_NEAREST.
Usage: patch-fpu.py <path/to/CGAL/FPU.h>  (idempotent)"""
import sys

path = sys.argv[1]
s = open(path).read()
if 'CGAL_IA_RTN_NO_BARRIERS' in s:
    print('already patched')
    sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

rep('# ifdef CGAL_HAS_SSE2\n', '# if defined(CGAL_HAS_SSE2) && !defined(__wasm__)\n')
rep('#  ifdef CGAL_HAS_SSE2\n', '#  if defined(CGAL_HAS_SSE2) && !defined(__wasm__)\n')

rep('''#if defined CGAL_FPU_HAS_EXCESS_PRECISION
#  define CGAL_IA_FORCE_TO_DOUBLE(x) CGAL::IA_force_to_double(x)
''', '''// In CGAL_ALWAYS_ROUND_TO_NEAREST mode the runtime and compile-time rounding modes are
// both round-to-nearest, and the directed widening that makes intervals valid is applied
// explicitly and asymmetrically by nextUp() (which is itself a bit-cast optimization
// barrier). The opacify/stop-cprop barriers exist solely to keep the compiler from folding
// or migrating ops across a hardware directed-rounding switch that never happens here, so
// they are correctness-neutral in this mode — and on Wasm IA_opacify forces a linear-memory
// spill+reload per interval op. CGAL_IA_RTN_NO_BARRIERS opts into dropping them.
#if defined(CGAL_ALWAYS_ROUND_TO_NEAREST) && defined(CGAL_IA_RTN_NO_BARRIERS)
#  define CGAL_IA_FORCE_TO_DOUBLE(x) (x)
#elif defined CGAL_FPU_HAS_EXCESS_PRECISION
#  define CGAL_IA_FORCE_TO_DOUBLE(x) CGAL::IA_force_to_double(x)
''')

rep('''#ifndef CGAL_IA_DONT_STOP_CONSTANT_PROPAGATION
#  define CGAL_IA_STOP_CPROP(x)    CGAL::IA_opacify(x)
''', '''#if defined(CGAL_ALWAYS_ROUND_TO_NEAREST) && defined(CGAL_IA_RTN_NO_BARRIERS)
#  define CGAL_IA_STOP_CPROP(x)    (x)
#elif !defined(CGAL_IA_DONT_STOP_CONSTANT_PROPAGATION)
#  define CGAL_IA_STOP_CPROP(x)    CGAL::IA_opacify(x)
''')

rep('''inline double IA_up(double d)
{
''', '''static inline int64_t as_i64(double x) {
  int64_t i;
  std::memcpy(&i, &x, sizeof i);
  return i;
}
static inline double as_f64(int64_t i) {
  double x;
  std::memcpy(&x, &i, sizeof x);
  return x;
}

// nextafter(x, +inf) for finite x: +1 ulp on the bit pattern for x >= +0, -1 for x < 0, with
// -0 mapped to the smallest positive denormal. Infinities and NaNs are assumed never to occur.
static inline double nextUp(double x) {
  int64_t u = as_i64(x);
  if (__builtin_expect((u << 1) == 0, 0)) {
    return as_f64(0x0000000000000001LL);
  }
  const int64_t sign = u < 0;
  return as_f64(u + (1 - sign * 2));
}

inline double IA_up(double d)
{
''')

rep('  return nextafter(d, std::numeric_limits<double>::infinity());\n',
    '  return nextUp(d);\n')

open(path, 'w').write(s)
print('patched', path)
