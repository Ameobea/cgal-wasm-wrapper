# CGAL 6.2 patch queue

These unified patches are the source of truth for the wrapper's CGAL customizations. `manifest.json` pins their order, patch checksums, and the pristine/patched SHA-256 of every affected header. Its archive entry records the CGAL 6.2 release tarball used to generate and test them.

| Patch | Purpose |
|---|---|
| `0001-wasm-fpu.patch` | Existing Wasm successor implementation and optional removal of hardware-rounding barriers. |
| `0002-remesh-planar-patches.patch` | Existing fallback for inconsistent planar-patch triangulations. |
| `0003-aabb-distance-tie.patch` | Existing exact-tie shortcut for AABB distance comparisons. |
| `0004-wasm-circumcenter-filter.patch` | Certified nearest-rounding polynomial filter for alpha-wrap's circumcenter-orientation predicate. |
| `0005-compare-angle-no-uncertain-branch.patch` | `Compare_angle_3` cosine overloads: branch-free sign-preserving-square comparison, so exactly perpendicular inputs no longer throw out of the interval filter (was every neighbor test in `remesh_planar_patches` on box-like meshes). |
| `0006-point-plane-distance-no-point-on-plane.patch` | `compare_squared_distance(point, plane, d2)` via the cross-multiplied plane equation instead of `plane.point()`, whose coefficient selection throws for near-axis-aligned planes. |

The first three are byte-for-byte conversions of the former Python patch scripts. The fourth is the previously tested/installed alpha-wrap patch from the Geoscript experiments. The FPU successor's existing finite-input assumption is retained; correcting it should be a separate reviewed numerical change.

The fifth and sixth are algebraically equivalent rewrites of the upstream comparisons (same result for every sign case, given a non-degenerate plane for the sixth); they only change which inputs the interval filter can certify without falling back to exact arithmetic. Unlike the fourth, they have no recorded exact-oracle or output-parity run.

## Apply and check

From `3DLibrary/wasm-library`:

```sh
just patch-deps       # apply every patch, or recognize an already-patched header
just check-patches   # verify the entire operation in temporary storage; no header writes
just test-patches    # needs libraries/downloads/CGAL-6.2.tar.xz
```

Equivalently, `python3 apply-patches.py [CGAL-release-root] [--check]`. The root contains `include/CGAL/`. `--only 0001-wasm-fpu.patch` selects a subset and may be repeated, which supports the isolated benchmark overlays.

All selected patches are applied in temporary storage and their output hashes are checked before any destination header is changed. Unexpected source contents or a modified patch are errors, not reasons to use fuzzy application. Already-patched headers are recognized by content, so a tree with only some of the patches applied (e.g. one patched by the former scripts) is completed in place. Per-file replacement is atomic; a filesystem failure can leave some files applied, and the next invocation recognizes them.

`just install` also touches Geotoy's Vite config, if present, to clear stale transformed JS after copying the Wasm/JS pair.

`just build` applies this queue and enables `CGAL_AW3_WASM_CIRCUMCENTER_FILTER` through CMake by default. Set `-DCGAL_WASM_CIRCUMCENTER_FILTER=OFF` when configuring an explicit control build. Compilation uses `-ffp-contract=off`. The predicate's own guards restrict it to Wasm with `CGAL_ALWAYS_ROUND_TO_NEAREST`; it rejects fast-math builds.

## Maintaining the queue

Edit the `.patch` files, not generated dependency headers. For an upstream upgrade, extract a fresh release, review/rebase the diffs, and update `manifest.json` with the new release/archive identity, each patch's checksum, and each target's before/after checksum. Review those regenerated headers before updating the expected hashes. Run the queue tests and relevant geometric/performance checks. The checksum checks intentionally force an explicit rebase when touched upstream files change.

The patch paths use the release-tarball layout. In the full upstream CGAL monorepo, package directory prefixes would be required. A maintained patch queue is sufficient for these isolated changes; a full CGAL fork is not required.

[Findings and numerical contract](FINDINGS.md) describe the alpha-wrap patch (`0004`) only. The experiment and comparison harnesses remain in the Geoscript checkout under `scripts/alpha-wrap-perf` and `scripts/simplify-perf`; they consume this queue rather than maintaining another copy.
