#!/usr/bin/env python3
"""Patches CGAL's AABB_traits_3.h so the statically filtered box/point compare_distance predicate
answers the exact-tie case (query point coincides with the current closest point, i.e. a point
already on the surface) itself. Otherwise it defers to the interval-filtered predicate, which
recomputes the bound distance with 1-ulp-widened intervals under CGAL_ALWAYS_ROUND_TO_NEAREST,
can never certify the tie, and falls back to exact arithmetic on every node visited (a 10-20x
slowdown of isotropic remeshing on flat geometry).
Usage: patch-aabb-distance-tie.py <path/to/CGAL/AABB_traits_3.h>  (idempotent)"""
import sys

path = sys.argv[1]
s = open(path).read()
if 'no box can be strictly closer' in s:
    print('already patched')
    sys.exit(0)

old = '''        if ((ssr < 1.11261183279326254436e-293) || (ssr > 2.80889552322236673473e+306)) {
          CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
          return Compare_distance_getter_3<GeomTraits, true, false>::compare_distance_object()(p, b, bound);
        }
'''
new = '''        // p coincides with bound: no box can be strictly closer, so the answer is exact here
        if (ssr == 0) {
          const bool inside = scx >= bxmin && scx <= bxmax && scy >= bymin && scy <= bymax &&
                              scz >= bzmin && scz <= bzmax;
          return inside ? CGAL::EQUAL : CGAL::LARGER;
        }
''' + old
assert s.count(old) == 1
s = s.replace(old, new)
open(path, 'w').write(s)
print('patched', path)
