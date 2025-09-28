#define CGAL_EIGEN3_ENABLED
#define CGAL_ALWAYS_ROUND_TO_NEAREST
// #define __wasm__
// #define __SSE2__
// #define __SSE4_1__

#include <CGAL/Dimension.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/region_growing.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/surface_Delaunay_remeshing.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_parameterization/Circular_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_authalic_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Error_code.h>
#include <CGAL/Surface_mesh_parameterization/IO/File_off.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>
#include <CGAL/boost/graph/properties.h>
#include <CGAL/subdivision_method_3.h>
#include <boost/foreach.hpp>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <fstream>
#include <igl/PI.h>
#include <iostream>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

// typedef CGAL::Simple_cartesian<double> Kernel;
// /\ using this kernel causes precision issues and all kinds of bugs.
// it needs the exact predicates kernel \/ to work properly.
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> SurfaceMesh;

template <typename T> intptr_t getVecDataPtr(std::vector<T> &vec) {
  return reinterpret_cast<intptr_t>(vec.data());
}

template <typename T>
emscripten::class_<std::vector<T>> register_vector_custom(const char *name) {
  typedef std::vector<T> VecType;

  void (VecType::*resize)(const size_t, const T &) = &VecType::resize;
  size_t (VecType::*size)() const = &VecType::size;
  return emscripten::class_<std::vector<T>>(name)
      .template constructor<>()
      .function("resize", resize)
      .function("size", size)
      .function("data", &getVecDataPtr<T>, emscripten::allow_raw_pointers());
}

class PolyMesh {
private:
  std::vector<double> vertices;
  std::vector<int> faces;

  using edge_descriptor = boost::graph_traits<SurfaceMesh>::edge_descriptor;
  using face_descriptor = boost::graph_traits<SurfaceMesh>::face_descriptor;
  using vertex_descriptor = boost::graph_traits<SurfaceMesh>::vertex_descriptor;

  // Reusable constrained edge map (persist across calls so remeshing can reuse
  // it)
  SurfaceMesh::Property_map<edge_descriptor, bool> constrained_emap;
  bool constrained_edge_map_initialized = false;

  void ensure_constrained_map() {
    if (!constrained_edge_map_initialized) {
      constrained_emap = mesh.add_property_map<edge_descriptor, bool>(
                                 "e:is_constrained", false)
                             .first;
      constrained_edge_map_initialized = true;
    }
  }

  // Mark borders as constrained
  void tag_borders() {
    ensure_constrained_map();
    for (auto e : edges(mesh)) {
      if (is_border(e, mesh)) {
        constrained_emap[e] = true;
      }
    }
  }

  void tag_edges(bool protect_borders, bool auto_sharp_edges,
                 double sharp_angle_threshold_degrees) {
    ensure_constrained_map();
    for (auto e : edges(mesh)) {
      constrained_emap[e] = false;
    }

    if (protect_borders) {
      tag_borders();
    }

    if (auto_sharp_edges) {
      PMP::detect_sharp_edges(mesh, sharp_angle_threshold_degrees,
                              constrained_emap);
    }
  }

public:
  SurfaceMesh mesh;
  Kernel::Point_3 sdf_p = Kernel::Point_3(0, 4, 0);
  PolyMesh() {}

  // TODO: should check to see if there's a native CGAL function for this
  void buildFromBuffers(const std::vector<float> &vertices,
                        const std::vector<uint32_t> &indices) {
    mesh.clear();

    if (vertices.size() % 3 != 0) {
      std::cout << "Error: Vertex buffer size must be divisible by 3 (x,y,z "
                   "components)"
                << std::endl;
      return;
    }

    if (indices.size() % 3 != 0) {
      std::cout
          << "Error: Index buffer size must be divisible by 3 (triangle faces)"
          << std::endl;
      return;
    }

    std::vector<SurfaceMesh::vertex_index> vertex_indices;
    vertex_indices.reserve(vertices.size() / 3);

    for (size_t i = 0; i < vertices.size(); i += 3) {
      Kernel::Point_3 point(static_cast<double>(vertices[i]),
                            static_cast<double>(vertices[i + 1]),
                            static_cast<double>(vertices[i + 2]));
      vertex_indices.push_back(mesh.add_vertex(point));
    }

    for (size_t i = 0; i < indices.size(); i += 3) {
      uint32_t i0 = indices[i];
      uint32_t i1 = indices[i + 1];
      uint32_t i2 = indices[i + 2];

      // TODO: crazy that we're allocating a vector each iteration here
      std::vector<SurfaceMesh::vertex_index> face_verts = {
          vertex_indices[i0], vertex_indices[i1], vertex_indices[i2]};

      auto f = mesh.add_face(face_verts);
      if (f == SurfaceMesh::null_face()) {
        throw std::runtime_error(
            "Error: Failed to add face to mesh. The input indices may "
            "contain duplicates or refer to non-existent vertices, incorrect "
            "winding order, or other issues.");
      }
    }
  }

  void triangulate_in_place() { PMP::triangulate_faces(mesh); }

  void catmull_smooth(int iterations) {
    CGAL::Subdivision_method_3::CatmullClark_subdivision(mesh, iterations);
  }

  void loop_smooth(int iterations) {
    CGAL::Subdivision_method_3::Loop_subdivision(mesh, iterations);
  }

  void dooSabin_smooth(int iterations) {
    CGAL::Subdivision_method_3::DooSabin_subdivision(mesh, iterations);
  }

  void sqrt_smooth(int iterations) {
    CGAL::Subdivision_method_3::Sqrt3_subdivision(mesh, iterations);
  }

  bool isTriangulated(const SurfaceMesh &mesh) {
    for (const auto &f : mesh.faces()) {
      if (mesh.degree(f) != 3) {
        return false;
      }
    }
    return true;
  }

  void maybeTriangulate() {
    if (!isTriangulated(mesh)) {
      triangulate_in_place();
    }
    mesh.collect_garbage();
  }

  std::vector<uint32_t> getIndices() {
    std::vector<uint32_t> indices;

    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end();
         ++faceIt) {
      auto halfedgeIt = mesh.halfedge(*faceIt);
      auto halfedgeIt2 = mesh.next(halfedgeIt);
      auto halfedgeIt3 = mesh.next(halfedgeIt2);
      auto vertexIt = mesh.target(halfedgeIt);
      auto vertexIt2 = mesh.target(halfedgeIt2);
      auto vertexIt3 = mesh.target(halfedgeIt3);
      indices.push_back(static_cast<uint32_t>(vertexIt.idx()));
      indices.push_back(static_cast<uint32_t>(vertexIt2.idx()));
      indices.push_back(static_cast<uint32_t>(vertexIt3.idx()));
    }

    return indices;
  }

  std::vector<float> getVertices() {
    std::vector<float> vertices;

    for (auto vertexIt = mesh.vertices_begin(); vertexIt != mesh.vertices_end();
         ++vertexIt) {
      auto point = mesh.point(*vertexIt);
      vertices.push_back(static_cast<float>(point.x()));
      vertices.push_back(static_cast<float>(point.y()));
      vertices.push_back(static_cast<float>(point.z()));
    }

    return vertices;
  }

  PolyMesh alphaWrap(double relative_alpha, double relative_offset) {
    if (mesh.is_empty()) {
      return PolyMesh();
    }

    CGAL::Bbox_3 bbox = PMP::bbox(mesh);
    const double diag_length =
        std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                  CGAL::square(bbox.ymax() - bbox.ymin()) +
                  CGAL::square(bbox.zmax() - bbox.zmin()));

    const double alpha = diag_length * relative_alpha;
    const double offset = diag_length * relative_offset;

    PolyMesh result;

    try {
      CGAL::alpha_wrap_3(mesh, alpha, offset, result.mesh);
    } catch (const std::exception &e) {
      std::cout << "Error during alpha wrapping: " << e.what() << std::endl;
    }

    return result;
  }

  void remesh_planar_patches(double max_angle_deg, double max_offset) {
    if (mesh.is_empty()) {
      return;
    }

    std::vector<std::size_t> region_ids(num_faces(mesh));
    // corner status of vertices
    std::vector<std::size_t> corner_id_map(num_vertices(mesh), -1);
    // mark edges at the boundary of regions
    std::vector<bool> ecm(num_edges(mesh), false);
    // normal of the supporting planes of the regions detected
    boost::vector_property_map<CGAL::Epick::Vector_3> normal_map;

    auto max_angle_rads = max_angle_deg * CGAL_PI / 180.0;
    auto cos_max_angle = std::cos(max_angle_rads);

    // detect planar regions in the mesh
    std::size_t nb_regions = PMP::region_growing_of_planes_on_faces(
        mesh, CGAL::make_random_access_property_map(region_ids),
        CGAL::parameters::cosine_of_maximum_angle(cos_max_angle)
            .region_primitive_map(normal_map)
            .maximum_distance(max_offset));

    // detect corner vertices on the boundary of planar regions
    std::size_t nb_corners = PMP::detect_corners_of_regions(
        mesh, CGAL::make_random_access_property_map(region_ids), nb_regions,
        CGAL::make_random_access_property_map(corner_id_map),
        CGAL::parameters::cosine_of_maximum_angle(cos_max_angle)
            .maximum_distance(max_offset)
            .edge_is_constrained_map(
                CGAL::make_random_access_property_map(ecm)));

    // run the remeshing algorithm using filled properties
    SurfaceMesh out;
    PMP::remesh_almost_planar_patches(
        mesh, out, nb_regions, nb_corners,
        CGAL::make_random_access_property_map(region_ids),
        CGAL::make_random_access_property_map(corner_id_map),
        CGAL::make_random_access_property_map(ecm),
        CGAL::parameters::patch_normal_map(normal_map));

    // dispose of old mesh and replace with new
    mesh = out;
    constrained_edge_map_initialized = false;
    out.clear();
    mesh.collect_garbage();
  }

  void isotropic_remesh(double target_edge_length, int iterations,
                        bool protect_borders, bool auto_sharp_edges,
                        double sharp_angle_threshold_degrees) {
    if (mesh.is_empty()) {
      return;
    }

    tag_edges(protect_borders, auto_sharp_edges, sharp_angle_threshold_degrees);

    // CGAL warns to pre-split long preserved edges to avoid infinite loops in
    // the algorithm
    PMP::split_long_edges(
        edges(mesh), target_edge_length, mesh,
        CGAL::parameters::edge_is_constrained_map(constrained_emap));

    PMP::isotropic_remeshing(mesh.faces(), target_edge_length, mesh,
                             CGAL::parameters::number_of_iterations(iterations)
                                 .edge_is_constrained_map(constrained_emap)
                                 .protect_constraints(true));
  }

  void delaunay_remesh(double target_edge_length, double facet_distance,
                       bool auto_sharp_edges,
                       double sharp_angle_threshold_degrees) {
    if (mesh.is_empty())
      return;

    auto outmesh = PMP::surface_Delaunay_remeshing(
        mesh, CGAL::parameters::protect_constraints(auto_sharp_edges)
                  .mesh_edge_size(target_edge_length)
                  .mesh_facet_distance(facet_distance)
                  .features_angle_bound(sharp_angle_threshold_degrees));

    mesh = std::move(outmesh);
    constrained_edge_map_initialized = false;
    mesh.collect_garbage();
  }
};

PolyMesh alphaWrapPointCloud(const std::vector<float> &vertices,
                             double relative_alpha, double relative_offset) {
  std::vector<Kernel::Point_3> points;

  if (vertices.size() % 3 != 0) {
    std::cout << "Error: Point cloud data must be a flat array of "
                 "[x,y,z,x,y,z,...] coordinates"
              << std::endl;
    return PolyMesh();
  }

  int num_points = vertices.size() / 3;
  points.reserve(num_points);

  for (int i = 0; i < num_points; i++) {
    double x = static_cast<double>(vertices[i * 3]);
    double y = static_cast<double>(vertices[i * 3 + 1]);
    double z = static_cast<double>(vertices[i * 3 + 2]);
    points.emplace_back(x, y, z);
  }

  if (points.empty()) {
    return PolyMesh();
  }

  CGAL::Bbox_3 bbox = CGAL::bbox_3(std::cbegin(points), std::cend(points));
  const double diag_length = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                       CGAL::square(bbox.ymax() - bbox.ymin()) +
                                       CGAL::square(bbox.zmax() - bbox.zmin()));
  const double alpha = diag_length * relative_alpha;
  const double offset = diag_length * relative_offset;

  PolyMesh result;

  try {
    CGAL::alpha_wrap_3(points, alpha, offset, result.mesh);
  } catch (const std::exception &e) {
    std::cout << "Error during point cloud alpha wrapping: " << e.what()
              << std::endl;
  }

  return result;
}

EMSCRIPTEN_BINDINGS(my_module) {
  register_vector_custom<float>("vector<float>");
  register_vector_custom<uint32_t>("vector<uint32_t>");

  emscripten::class_<PolyMesh>("PolyMesh")
      .constructor<>()
      .function("buildFromBuffers", &PolyMesh::buildFromBuffers,
                emscripten::allow_raw_pointers())
      .function("getIndices", &PolyMesh::getIndices,
                emscripten::allow_raw_pointers())
      .function("getVertices", &PolyMesh::getVertices,
                emscripten::allow_raw_pointers())
      .function("catmull_smooth", &PolyMesh::catmull_smooth,
                emscripten::allow_raw_pointers())
      .function("loop_smooth", &PolyMesh::loop_smooth,
                emscripten::allow_raw_pointers())
      .function("sqrt_smooth", &PolyMesh::sqrt_smooth,
                emscripten::allow_raw_pointers())
      .function("dooSabin_smooth", &PolyMesh::dooSabin_smooth,
                emscripten::allow_raw_pointers())
      .function("alphaWrap", &PolyMesh::alphaWrap,
                emscripten::allow_raw_pointers())
      .function("remesh_planar_patches", &PolyMesh::remesh_planar_patches,
                emscripten::allow_raw_pointers())
      .function("isotropic_remesh", &PolyMesh::isotropic_remesh,
                emscripten::allow_raw_pointers())
      .function("delaunay_remesh", &PolyMesh::delaunay_remesh,
                emscripten::allow_raw_pointers())
      .function("maybe_triangulate", &PolyMesh::maybeTriangulate,
                emscripten::allow_raw_pointers());

  emscripten::function("alphaWrapPointCloud", &alphaWrapPointCloud,
                       emscripten::allow_raw_pointers());
}
