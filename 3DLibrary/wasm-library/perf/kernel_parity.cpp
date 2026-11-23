// Parity harness for the OTHER CGAL builtins affected by the IA-barrier change:
//  - 2D Constrained Delaunay Triangulation + Delaunay mesh refinement (EPICK predicates)
//  - 2D polygon boolean set ops (EPECK / Lazy_exact_nt exact constructions)
// Prints a fingerprint of each output; run baseline vs nobarriers and diff the fingerprints.
// Mirrors the kernel/typedef choices in main.cpp.

#define CGAL_ALWAYS_ROUND_TO_NEAREST

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/mark_domain_in_triangulation.h>

#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Polygon_with_holes_2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Triangulation_vertex_base_2<Kernel> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<Kernel> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> TDS;
typedef CGAL::No_constraint_intersection_requiring_constructions_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS, Itag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CGAL::Exact_predicates_exact_constructions_kernel EK;
typedef CGAL::Polygon_2<EK> EK_Polygon_2;
typedef CGAL::Polygon_with_holes_2<EK> EK_PWH;
typedef CGAL::Polygon_set_2<EK> EK_Polygon_set;

struct FNV {
  uint64_t h = 1469598103934665603ULL;
  void mix(uint64_t x) {
    for (int b = 0; b < 8; b++) { h ^= (x & 0xff); h *= 1099511628211ULL; x >>= 8; }
  }
  void d(double v) { mix((uint64_t)std::llround(v * 1e6)); }
};

static uint64_t test_cdt_refine() {
  // A square with a square hole, then refine with size/angle criteria (Steiner points are
  // EPICK inexact constructions -> coordinates depend on predicate-driven structure).
  CDT cdt;
  auto add_loop = [&](std::vector<std::pair<double, double>> pts) {
    std::vector<CDT::Vertex_handle> h;
    for (auto &p : pts) h.push_back(cdt.insert(CDT::Point(p.first, p.second)));
    for (size_t i = 0; i < h.size(); i++)
      cdt.insert_constraint(h[i], h[(i + 1) % h.size()]);
  };
  add_loop({{0, 0}, {10, 0}, {10, 10}, {0, 10}});
  add_loop({{3, 3}, {3, 7}, {7, 7}, {7, 3}});
  CGAL::mark_domain_in_triangulation(cdt);
  CGAL::refine_Delaunay_mesh_2(
      cdt, CGAL::parameters::domain_is_initialized(true).criteria(Criteria(0.125, 0.5)));
  FNV f;
  size_t nv = 0, nf = 0;
  for (auto v = cdt.finite_vertices_begin(); v != cdt.finite_vertices_end(); ++v) {
    f.d(v->point().x()); f.d(v->point().y()); nv++;
  }
  for (auto fc = cdt.finite_faces_begin(); fc != cdt.finite_faces_end(); ++fc)
    if (fc->is_in_domain()) nf++;
  f.mix(nv); f.mix(nf);
  fprintf(stderr, "  cdt_refine: verts=%zu faces=%zu\n", nv, nf);
  return f.h;
}

static uint64_t test_boolean() {
  // Union/intersection/difference of two overlapping rotated polygons -> exact-construction
  // intersection vertices (Lazy_exact_nt). Fingerprint all output coords.
  auto make_poly = [](double cx, double cy, double r, int n, double rot) {
    EK_Polygon_2 p;
    for (int i = 0; i < n; i++) {
      double a = rot + 2 * M_PI * i / n;
      p.push_back(EK::Point_2(cx + r * std::cos(a), cy + r * std::sin(a)));
    }
    if (p.orientation() == CGAL::CLOCKWISE) p.reverse_orientation();
    return p;
  };
  EK_Polygon_set A, B;
  A.insert(make_poly(0, 0, 5, 7, 0.3));
  B.insert(make_poly(3, 1, 4, 5, 1.1));
  FNV f;
  for (int op = 0; op < 3; op++) {
    EK_Polygon_set R(A);
    if (op == 0) R.join(B);
    else if (op == 1) R.intersection(B);
    else R.difference(B);
    std::vector<EK_PWH> out;
    R.polygons_with_holes(std::back_inserter(out));
    size_t nc = 0;
    for (auto &pwh : out) {
      if (pwh.is_unbounded()) continue;
      for (auto it = pwh.outer_boundary().vertices_begin();
           it != pwh.outer_boundary().vertices_end(); ++it) {
        f.d(CGAL::to_double(it->x())); f.d(CGAL::to_double(it->y())); nc++;
      }
      for (auto hh = pwh.holes_begin(); hh != pwh.holes_end(); ++hh)
        for (auto it = hh->vertices_begin(); it != hh->vertices_end(); ++it) {
          f.d(CGAL::to_double(it->x())); f.d(CGAL::to_double(it->y())); nc++;
        }
    }
    f.mix(nc);
    fprintf(stderr, "  boolean op%d: out_coords=%zu\n", op, nc);
  }
  return f.h;
}

int main() {
  uint64_t a = test_cdt_refine();
  uint64_t b = test_boolean();
  printf("cdt=%016llx boolean=%016llx\n", (unsigned long long)a, (unsigned long long)b);
  return 0;
}
