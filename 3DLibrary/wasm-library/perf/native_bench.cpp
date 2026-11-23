// Native repro of the Wasm alpha_wrap path for fast iteration + output-parity checks.
// Mirrors the Wasm binding `alphaWrapPointCloud` (main.cpp) and its EPICK kernel +
// CGAL_ALWAYS_ROUND_TO_NEAREST config. Generates deterministic point clouds so runs are
// reproducible and output can be diffed byte-for-byte across code variants.
//
// Usage: ./native_bench <case> <npoints> <relalpha_inv> <reloffset_inv> <reps> [dump.bin]
//   case: torus_shell | sphere_shell | noise_shell
//   relalpha_inv/reloffset_inv: pass e.g. 100 for alpha=1/100
// Prints per-rep wall time (ms) and a hash of the output mesh. Optional dump writes raw
// float verts + uint32 indices for cross-variant diffing.

#define CGAL_ALWAYS_ROUND_TO_NEAREST

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> SurfaceMesh;

// Deterministic LCG so point clouds are identical across builds/runs.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double uniform(double lo, double hi) {
    return lo + (hi - lo) * ((next() >> 11) * (1.0 / 9007199254740992.0));
  }
};

static double dist_to_circle(double x, double y, double z, double r) {
  double a = std::sqrt(x * x + z * z) - r;
  return std::sqrt(y * y + a * a);
}

// Cheap value-noise-ish field for the noise_shell case (not CGAL-relevant, just to vary the
// point distribution); deterministic.
static double hash_noise(double x, double y, double z) {
  double v = std::sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
  return v - std::floor(v);
}

static std::vector<float> gen_points(const std::string &which, int n, uint64_t seed) {
  Rng rng(seed);
  std::vector<float> pts;
  pts.reserve(n * 3);
  if (which == "torus_shell") {
    const double radius = 8.0, band = 2.0, R = radius * 1.1;
    while ((int)(pts.size() / 3) < n) {
      double x = rng.uniform(-R, R), y = rng.uniform(-R, R), z = rng.uniform(-R, R);
      if (dist_to_circle(x, y, z, radius) < band) {
        pts.push_back((float)x);
        pts.push_back((float)y);
        pts.push_back((float)z);
      }
    }
  } else if (which == "sphere_shell") {
    const double radius = 12.0;
    while ((int)(pts.size() / 3) < n) {
      double x = rng.uniform(-radius, radius), y = rng.uniform(-radius, radius),
             z = rng.uniform(-radius, radius);
      double l = std::sqrt(x * x + y * y + z * z);
      if (l < radius && l > radius * 0.82) {
        pts.push_back((float)x);
        pts.push_back((float)y);
        pts.push_back((float)z);
      }
    }
  } else { // noise_shell
    const double radius = 12.0;
    while ((int)(pts.size() / 3) < n) {
      double x = rng.uniform(-radius, radius), y = rng.uniform(-radius, radius),
             z = rng.uniform(-radius, radius);
      if (std::sqrt(x * x + y * y + z * z) >= radius) continue;
      double nz = hash_noise(x * 0.0283, y * 0.0283, z * 0.0283);
      if ((nz > 0.20 && nz < 0.24) || (nz > 0.48 && nz < 0.52) || (nz > 0.76 && nz < 0.80)) {
        pts.push_back((float)x);
        pts.push_back((float)y);
        pts.push_back((float)z);
      }
    }
  }
  return pts;
}

// FNV-1a over the rounded output coords + indices: stable fingerprint for parity checks.
static uint64_t fingerprint(const std::vector<float> &v, const std::vector<uint32_t> &idx) {
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](uint64_t x) {
    for (int b = 0; b < 8; b++) {
      h ^= (x & 0xff);
      h *= 1099511628211ULL;
      x >>= 8;
    }
  };
  for (float f : v) {
    // round to ~1e-5 to be robust to last-bit noise while still catching real divergence
    int64_t q = (int64_t)std::llround((double)f * 100000.0);
    mix((uint64_t)q);
  }
  for (uint32_t i : idx) mix(i);
  mix(v.size());
  mix(idx.size());
  return h;
}

int main(int argc, char **argv) {
  std::string which = argc > 1 ? argv[1] : "torus_shell";
  int n = argc > 2 ? std::atoi(argv[2]) : 1550;
  double ralpha = 1.0 / (argc > 3 ? std::atof(argv[3]) : 100.0);
  double roffset = 1.0 / (argc > 4 ? std::atof(argv[4]) : 100.0);
  int reps = argc > 5 ? std::atoi(argv[5]) : 5;
  const char *dump = argc > 6 ? argv[6] : nullptr;

  std::vector<float> verts = gen_points(which, n, 0xC0FFEEULL);
  fprintf(stderr, "case=%s npoints=%d alpha=1/%g offset=1/%g reps=%d\n", which.c_str(), n,
          1.0 / ralpha, 1.0 / roffset, reps);

  std::vector<Kernel::Point_3> points;
  points.reserve(verts.size() / 3);
  for (size_t i = 0; i < verts.size(); i += 3)
    points.emplace_back(verts[i], verts[i + 1], verts[i + 2]);

  CGAL::Bbox_3 bbox = CGAL::bbox_3(points.begin(), points.end());
  const double diag = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                CGAL::square(bbox.ymax() - bbox.ymin()) +
                                CGAL::square(bbox.zmax() - bbox.zmin()));
  const double alpha = diag * ralpha, offset = diag * roffset;

  // Optional voxel-grid downsample to ~AW_DOWNSAMPLE * alpha spacing (keep 1 pt/cell).
  // alpha_wrap cannot resolve detail finer than alpha, so points much closer than alpha are
  // redundant; dropping them cuts oracle+triangulation work with ~unchanged output.
  const char *dsenv = std::getenv("AW_DOWNSAMPLE");
  double dsfrac = dsenv ? std::atof(dsenv) : 0.0;
  if (dsfrac > 0.0) {
    const double cell = alpha * dsfrac;
    std::unordered_set<uint64_t> seen;
    std::vector<Kernel::Point_3> kept;
    kept.reserve(points.size());
    auto key = [&](const Kernel::Point_3 &p) {
      // pack signed cell indices (biased to unsigned) into 21 bits each — collision-free
      // for |coord/cell| < 2^20, which holds for these bounded clouds.
      uint64_t ix = (uint64_t)((int64_t)std::floor(p.x() / cell) + 1048576);
      uint64_t iy = (uint64_t)((int64_t)std::floor(p.y() / cell) + 1048576);
      uint64_t iz = (uint64_t)((int64_t)std::floor(p.z() / cell) + 1048576);
      return (ix & 0x1FFFFF) | ((iy & 0x1FFFFF) << 21) | ((iz & 0x1FFFFF) << 42);
    };
    for (auto &p : points)
      if (seen.insert(key(p)).second) kept.push_back(p);
    fprintf(stderr, "downsample @%.2f*alpha (cell=%.4g): %zu -> %zu points\n", dsfrac, cell,
            points.size(), kept.size());
    points.swap(kept);
  }

  double best = 1e30, sum = 0;
  uint64_t fp = 0;
  std::vector<float> ov;
  std::vector<uint32_t> oi;
  for (int r = 0; r < reps; r++) {
    SurfaceMesh out;
    auto t0 = std::chrono::high_resolution_clock::now();
    CGAL::alpha_wrap_3(points, alpha, offset, out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    best = std::min(best, ms);
    sum += ms;

    ov.clear();
    oi.clear();
    for (auto v : out.vertices()) {
      auto p = out.point(v);
      ov.push_back((float)p.x());
      ov.push_back((float)p.y());
      ov.push_back((float)p.z());
    }
    for (auto f : out.faces()) {
      auto h = out.halfedge(f);
      oi.push_back((uint32_t)out.target(h).idx());
      oi.push_back((uint32_t)out.target(out.next(h)).idx());
      oi.push_back((uint32_t)out.target(out.next(out.next(h))).idx());
    }
    fp = fingerprint(ov, oi);
    fprintf(stderr, "  rep %d: %.1f ms  verts=%zu faces=%zu fp=%016llx\n", r, ms,
            ov.size() / 3, oi.size() / 3, (unsigned long long)fp);
  }
  fprintf(stderr, "best=%.1fms mean=%.1fms verts=%zu faces=%zu fp=%016llx\n", best,
          sum / reps, ov.size() / 3, oi.size() / 3, (unsigned long long)fp);
  printf("%.2f %016llx %zu %zu\n", best, (unsigned long long)fp, ov.size() / 3,
         oi.size() / 3);

  if (dump) {
    FILE *f = fopen(dump, "wb");
    uint64_t nv = ov.size(), ni = oi.size();
    fwrite(&nv, 8, 1, f);
    fwrite(&ni, 8, 1, f);
    fwrite(ov.data(), 4, nv, f);
    fwrite(oi.data(), 4, ni, f);
    fclose(f);
    fprintf(stderr, "dumped %s\n", dump);
  }
  return 0;
}
