// Smoke-test the REAL app artifact (build/index.{js,wasm}) via embind in Node:
// build a torus-shell point cloud, run alphaWrapPointCloud, sanity-check the output mesh.
import { CGAL as factory } from "../build/index.js";

function torusShell(n) {
  // deterministic-ish; matches native_bench torus_shell distribution roughly
  let s = 0xc0ffee >>> 0;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const R = 8, band = 2, B = R * 1.1, out = [];
  while (out.length / 3 < n) {
    const x = (rnd() * 2 - 1) * B, y = (rnd() * 2 - 1) * B, z = (rnd() * 2 - 1) * B;
    const a = Math.sqrt(x * x + z * z) - R;
    if (Math.sqrt(y * y + a * a) < band) out.push(x, y, z);
  }
  return out;
}

const M = await factory();
const pts = torusShell(1550);
const vec = new M.vector$float$();
vec.resize(pts.length, 0);
const ptr = vec.data();
M.HEAPF32.subarray(ptr / 4, ptr / 4 + pts.length).set(pts);

const t0 = performance.now();
const wrapped = M.alphaWrapPointCloud(vec, 1 / 100, 1 / 100);
const t1 = performance.now();
const verts = wrapped.getVertices();
const idx = wrapped.getIndices();
const nv = verts.size(), nf = idx.size() / 3;
console.log(`real-artifact alphaWrapPointCloud: ${(t1 - t0).toFixed(0)} ms  verts=${nv / 3} faces=${nf}`);
// sanity: closed-ish triangle mesh, Euler-ish, no degenerate empty
const ok = nv > 0 && nf > 100 && idx.size() % 3 === 0;
console.log(ok ? "SMOKE OK" : "SMOKE FAIL");
vec.delete();
wrapped.delete();
process.exit(ok ? 0 : 1);
