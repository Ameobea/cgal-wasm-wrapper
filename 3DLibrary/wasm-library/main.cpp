#define CGAL_EIGEN3_ENABLED
#define CGAL_ALWAYS_ROUND_TO_NEAREST
// #define __wasm__
// #define __SSE2__
// #define __SSE4_1__

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
#include <CGAL/Surface_mesh.h>
#include <CGAL/Multipolygon_with_holes_2.h>
#include <CGAL/alpha_wrap_2.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>
#include <CGAL/boost/graph/properties.h>
#include <CGAL/subdivision_method_3.h>
#include <emscripten.h>
#include <emscripten/bind.h>
#include <iostream>
#include <string>
#include <vector>

#include <CGAL/property_map.h>

// 2D Constrained Delaunay Triangulation
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/mark_domain_in_triangulation.h>

// 2D Boolean Set Operations (exact-arithmetic backend)
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_set_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <map>

namespace PMP = CGAL::Polygon_mesh_processing;

// typedef CGAL::Simple_cartesian<double> Kernel;
// /\ using this kernel causes precision issues and all kinds of bugs.
// it needs the exact predicates kernel \/ to work properly.
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> SurfaceMesh;

template <typename T> intptr_t getVecDataPtr(std::vector<T> &vec) {
  return reinterpret_cast<intptr_t>(vec.data());
}

static std::vector<Kernel::Point_3> to_points(const std::vector<float> &flat) {
  std::vector<Kernel::Point_3> points;
  points.reserve(flat.size() / 3);
  for (size_t i = 0; i + 2 < flat.size(); i += 3) {
    points.emplace_back(flat[i], flat[i + 1], flat[i + 2]);
  }
  return points;
}

static auto alpha_wrap_params(bool manifold,
                              const std::vector<Kernel::Point_3> &seeds) {
  return CGAL::parameters::do_enforce_manifoldness(manifold).seed_points(seeds);
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
  std::string last_error_;

  using edge_descriptor = boost::graph_traits<SurfaceMesh>::edge_descriptor;
  using face_descriptor = boost::graph_traits<SurfaceMesh>::face_descriptor;
  using vertex_descriptor = boost::graph_traits<SurfaceMesh>::vertex_descriptor;

  // Reusable constrained edge map (persist across calls so remeshing can reuse
  // it)
  SurfaceMesh::Property_map<edge_descriptor, bool> constrained_emap;
  bool constrained_edge_map_initialized = false;

  bool set_error(const std::string &message) {
    last_error_ = message;
    return false;
  }

  void clear_error() { last_error_.clear(); }

  // Attribute seams and other edges the caller needs kept as edges of the output.
  SurfaceMesh::Property_map<edge_descriptor, bool> feature_emap;

  void ensure_feature_map() {
    feature_emap =
        mesh.add_property_map<edge_descriptor, bool>("e:is_feature", false).first;
  }

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

  // A zero-area face (boolean output has them) has no normal, so anything classifying edges
  // or faces by angle misreads it: isotropic remeshing tags its edges sharp and can then only
  // subdivide it into a chain of more of them.
  void remove_degenerate_faces() {
    ensure_feature_map();
    PMP::remove_almost_degenerate_faces(
        mesh, CGAL::parameters::cap_threshold(std::cos(179.9 * CGAL_PI / 180.0))
                  .needle_threshold(1e5)
                  .edge_is_constrained_map(feature_emap));
    mesh.collect_garbage();
  }

  // Splits each region into its components connected across non-feature edges, so feature
  // edges always end up on region boundaries. Returns the new region count.
  std::size_t split_regions_at_features(
      std::vector<std::size_t> &region_ids,
      boost::vector_property_map<CGAL::Epick::Vector_3> &normal_map) {
    const std::size_t unset = -1;
    std::vector<std::size_t> out(region_ids.size(), unset);
    boost::vector_property_map<CGAL::Epick::Vector_3> out_normals;
    std::size_t count = 0;
    std::vector<face_descriptor> stack;
    for (auto seed : mesh.faces()) {
      if (out[seed] != unset) {
        continue;
      }
      out_normals[count] = normal_map[region_ids[seed]];
      out[seed] = count;
      stack.push_back(seed);
      while (!stack.empty()) {
        auto f = stack.back();
        stack.pop_back();
        for (auto h : halfedges_around_face(halfedge(f, mesh), mesh)) {
          auto g = face(opposite(h, mesh), mesh);
          if (g == SurfaceMesh::null_face() || out[g] != unset ||
              region_ids[g] != region_ids[f] || feature_emap[edge(h, mesh)]) {
            continue;
          }
          out[g] = count;
          stack.push_back(g);
        }
      }
      count += 1;
    }
    region_ids = std::move(out);
    normal_map = out_normals;
    return count;
  }

public:
  SurfaceMesh mesh;
  Kernel::Point_3 sdf_p = Kernel::Point_3(0, 4, 0);
  PolyMesh() {}

  // TODO: should check to see if there's a native CGAL function for this
  bool buildFromBuffers(const std::vector<float> &vertices,
                        const std::vector<uint32_t> &indices) {
    mesh.clear();
    clear_error();

    if (vertices.size() % 3 != 0) {
      std::cout << "Error: Vertex buffer size must be divisible by 3 (x,y,z "
                   "components)"
                << std::endl;
      return set_error(
          "Vertex buffer size must be divisible by 3 (x,y,z components)");
    }

    if (indices.size() % 3 != 0) {
      std::cout
          << "Error: Index buffer size must be divisible by 3 (triangle faces)"
          << std::endl;
      return set_error(
          "Index buffer size must be divisible by 3 (triangle faces)");
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
        std::string error_msg =
            "Failed to add face to mesh. The input indices may contain "
            "duplicates or refer to non-existent vertices, incorrect winding "
            "order, or other issues.";
        std::cout << "Error: " << error_msg << std::endl;
        return set_error(error_msg);
      }
    }

    return true;
  }

  // `pairs` holds two vertex indices per edge, in `buildFromBuffers` numbering.
  void set_feature_edges(const std::vector<uint32_t> &pairs) {
    ensure_feature_map();
    for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
      auto h = mesh.halfedge(SurfaceMesh::Vertex_index(pairs[i]),
                             SurfaceMesh::Vertex_index(pairs[i + 1]));
      if (h != SurfaceMesh::null_halfedge()) {
        feature_emap[mesh.edge(h)] = true;
      }
    }
  }

  std::string getLastError() const { return last_error_; }

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

  PolyMesh alphaWrap(double relative_alpha, double relative_offset,
                     bool manifold, const std::vector<float> &seeds) {
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
    const auto seed_points = to_points(seeds);

    try {
      CGAL::alpha_wrap_3(mesh, alpha, offset, result.mesh,
                         alpha_wrap_params(manifold, seed_points));
    } catch (const std::exception &e) {
      std::cout << "Error during alpha wrapping: " << e.what() << std::endl;
    }

    return result;
  }

  // LeastSquares: refit each region's plane by PCA as it grows (CGAL < 6 behavior) instead of
  // keeping the seed face's plane
  template <class LeastSquares>
  bool remesh_planar_patches_with(double max_angle_deg, double max_offset) {
    clear_error();
    if (mesh.is_empty()) {
      return true;
    }
    try {
      remove_degenerate_faces();
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
              .maximum_distance(max_offset)
              .least_squares_fitting(LeastSquares()));
      nb_regions = split_regions_at_features(region_ids, normal_map);

      // detect corner vertices on the boundary of planar regions
      std::size_t nb_corners = PMP::detect_corners_of_regions(
          mesh, CGAL::make_random_access_property_map(region_ids), nb_regions,
          CGAL::make_random_access_property_map(corner_id_map),
          CGAL::parameters::cosine_of_maximum_angle(cos_max_angle)
              .maximum_distance(max_offset)
              .least_squares_fitting(LeastSquares())
              .edge_is_constrained_map(
                  CGAL::make_random_access_property_map(ecm)));

      // returns false both when some patches were kept as-is (output still
      // valid) and when the remeshed patches could not be assembled (output
      // left empty)
      SurfaceMesh out;
      PMP::remesh_almost_planar_patches(
          mesh, out, nb_regions, nb_corners,
          CGAL::make_random_access_property_map(region_ids),
          CGAL::make_random_access_property_map(corner_id_map),
          CGAL::make_random_access_property_map(ecm),
          CGAL::parameters::patch_normal_map(normal_map));
      if (out.is_empty()) {
        return set_error(
            "CGAL could not assemble the remeshed planar patches into a valid "
            "mesh. This is usually caused by sliver triangles or near-coincident "
            "vertices at the scale of float precision, often left behind by "
            "boolean operations; try `simplify` first or a different "
            "`max_angle_deg`.");
      }

      mesh = out;
      constrained_edge_map_initialized = false;
      mesh.collect_garbage();
      return true;
    } catch (const std::exception &e) {
      return set_error(std::string("CGAL error in remesh_planar_patches: ") +
                       e.what());
    }
  }

  bool remesh_planar_patches(double max_angle_deg, double max_offset,
                             bool least_squares) {
    return least_squares
               ? remesh_planar_patches_with<CGAL::Tag_true>(max_angle_deg, max_offset)
               : remesh_planar_patches_with<CGAL::Tag_false>(max_angle_deg, max_offset);
  }

  void isotropic_remesh(double target_edge_length, int iterations,
                        bool protect_borders, bool auto_sharp_edges,
                        double sharp_angle_threshold_degrees) {
    if (mesh.is_empty()) {
      return;
    }

    remove_degenerate_faces();

    tag_edges(protect_borders, auto_sharp_edges, sharp_angle_threshold_degrees);
    for (auto e : edges(mesh)) {
      if (feature_emap[e]) {
        constrained_emap[e] = true;
      }
    }

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

    remove_degenerate_faces();
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
                             double relative_alpha, double relative_offset,
                             bool manifold, const std::vector<float> &seeds) {
  if (vertices.size() % 3 != 0) {
    std::cout << "Error: Point cloud data must be a flat array of "
                 "[x,y,z,x,y,z,...] coordinates"
              << std::endl;
    return PolyMesh();
  }

  const auto points = to_points(vertices);
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
  const auto seed_points = to_points(seeds);

  try {
    CGAL::alpha_wrap_3(points, alpha, offset, result.mesh,
                       alpha_wrap_params(manifold, seed_points));
  } catch (const std::exception &e) {
    std::cout << "Error during point cloud alpha wrapping: " << e.what()
              << std::endl;
  }

  return result;
}

typedef CGAL::Triangulation_vertex_base_2<Kernel> CDT2D_Vb;
typedef CGAL::Constrained_triangulation_face_base_2<Kernel> CDT2D_Fb;
typedef CGAL::Triangulation_data_structure_2<CDT2D_Vb, CDT2D_Fb> CDT2D_TDS;
typedef CGAL::No_constraint_intersection_requiring_constructions_tag CDT2D_Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, CDT2D_TDS,
                                                   CDT2D_Itag>
    CDT2D;

class CDT2DResult {
private:
  std::vector<float> vertices_;         // x, y pairs for output vertices
  std::vector<uint32_t> indices_;       // triangle indices (3 per triangle)
  std::vector<int32_t> vertex_mapping_; // input vtx idx -> output vtx idx
  bool success_;
  std::string error_;

public:
  CDT2DResult() : success_(false), error_("Not initialized") {}

  CDT2DResult(std::vector<float> vertices, std::vector<uint32_t> indices,
              std::vector<int32_t> vertex_mapping)
      : vertices_(std::move(vertices)), indices_(std::move(indices)),
        vertex_mapping_(std::move(vertex_mapping)), success_(true) {}

  CDT2DResult(std::string error) : success_(false), error_(std::move(error)) {}

  bool success() const { return success_; }
  std::string getError() const { return error_; }

  std::vector<float> getVertices() const { return vertices_; }
  std::vector<uint32_t> getIndices() const { return indices_; }
  std::vector<int32_t> getVertexMapping() const { return vertex_mapping_; }

  size_t getVertexCount() const { return vertices_.size() / 2; }
  size_t getTriangleCount() const { return indices_.size() / 3; }
};

CDT2DResult triangulatePolygon2D(const std::vector<float> &input_vertices) {
  // Input: flat array of 2D points [x0, y0, x1, y1, ...] forming a closed
  // polygon
  if (input_vertices.size() < 6) {
    return CDT2DResult("Need at least 3 vertices (6 floats) to form a polygon");
  }
  if (input_vertices.size() % 2 != 0) {
    return CDT2DResult("Input must have even number of floats (x, y pairs)");
  }

  const size_t num_input_verts = input_vertices.size() / 2;

  CDT2D cdt;
  std::vector<CDT2D::Vertex_handle> handles(num_input_verts);

  for (size_t i = 0; i < num_input_verts; ++i) {
    const double x = static_cast<double>(input_vertices[2 * i]);
    const double y = static_cast<double>(input_vertices[2 * i + 1]);
    handles[i] = cdt.insert(CDT2D::Point(x, y));
  }

  try {
    for (size_t i = 0; i < num_input_verts; ++i) {
      const size_t next = (i + 1) % num_input_verts;
      // Skip zero-length edges (coincident vertices)
      if (handles[i] != handles[next]) {
        cdt.insert_constraint(handles[i], handles[next]);
      }
    }
  } catch (const CDT2D::Intersection_of_constraints_exception &) {
    return CDT2DResult("Self-intersecting polygon detected");
  } catch (const std::exception &e) {
    return CDT2DResult(std::string("Error inserting constraints: ") + e.what());
  }

  if (cdt.dimension() < 2) {
    return CDT2DResult("Degenerate polygon - all points are collinear");
  }

  // Mark interior faces using CGAL's flood-fill algorithm
  std::unordered_map<CDT2D::Face_handle, bool> in_domain_map;
  boost::associative_property_map<std::unordered_map<CDT2D::Face_handle, bool>>
      in_domain(in_domain_map);
  CGAL::mark_domain_in_triangulation(cdt, in_domain);

  // Build output vertices and handle-to-index mapping
  std::vector<float> out_vertices;
  std::map<CDT2D::Vertex_handle, uint32_t> vh_to_idx;

  for (auto vh = cdt.finite_vertices_begin(); vh != cdt.finite_vertices_end();
       ++vh) {
    vh_to_idx[vh] = static_cast<uint32_t>(out_vertices.size() / 2);
    out_vertices.push_back(static_cast<float>(vh->point().x()));
    out_vertices.push_back(static_cast<float>(vh->point().y()));
  }

  // Build vertex mapping: input index -> output index
  // Handles coincident vertices automatically (they share the same handle)
  std::vector<int32_t> vertex_mapping(num_input_verts);
  for (size_t i = 0; i < num_input_verts; ++i) {
    vertex_mapping[i] = static_cast<int32_t>(vh_to_idx[handles[i]]);
  }

  // Extract interior triangles
  std::vector<uint32_t> out_indices;
  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end();
       ++fit) {
    if (in_domain_map[fit]) {
      out_indices.push_back(vh_to_idx[fit->vertex(0)]);
      out_indices.push_back(vh_to_idx[fit->vertex(1)]);
      out_indices.push_back(vh_to_idx[fit->vertex(2)]);
    }
  }

  if (out_indices.empty()) {
    return CDT2DResult("No interior triangles found - polygon may be "
                       "degenerate or have incorrect winding");
  }

  return CDT2DResult(std::move(out_vertices), std::move(out_indices),
                     std::move(vertex_mapping));
}

// Multi-subpath / refining CDT.  Uses `Delaunay_mesh_face_base_2` so that
// `is_in_domain()` is stored on each face by `mark_domain_in_triangulation`
// and so `Delaunay_mesher_2` can refine.
typedef CGAL::Triangulation_vertex_base_2<Kernel> CDT2DH_Vb;
typedef CGAL::Delaunay_mesh_face_base_2<Kernel> CDT2DH_Fb;
typedef CGAL::Triangulation_data_structure_2<CDT2DH_Vb, CDT2DH_Fb> CDT2DH_TDS;
typedef CGAL::No_constraint_intersection_requiring_constructions_tag
    CDT2DH_Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, CDT2DH_TDS,
                                                   CDT2DH_Itag>
    CDT2DH;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT2DH> CDT2DH_Criteria;

CDT2DResult
triangulatePolygon2DWithHoles(const std::vector<float> &input_vertices,
                              const std::vector<uint32_t> &subpath_lengths,
                              float max_edge_len, float min_angle_bound,
                              bool refine,
                              const std::vector<float> &interior_points) {
  if (input_vertices.size() % 2 != 0) {
    return CDT2DResult("Input must have even number of floats (x, y pairs)");
  }
  if (interior_points.size() % 2 != 0) {
    return CDT2DResult("interior_points must have even number of floats (x, y pairs)");
  }
  if (subpath_lengths.empty()) {
    return CDT2DResult("Need at least one subpath");
  }

  const size_t num_input_verts = input_vertices.size() / 2;
  size_t total_from_subpaths = 0;
  for (auto len : subpath_lengths) {
    if (len < 3) {
      return CDT2DResult("Each subpath must have at least 3 vertices");
    }
    total_from_subpaths += len;
  }
  if (total_from_subpaths != num_input_verts) {
    return CDT2DResult(
        "Sum of subpath_lengths does not match the vertex count");
  }

  CDT2DH cdt;
  std::vector<CDT2DH::Vertex_handle> handles(num_input_verts);

  for (size_t i = 0; i < num_input_verts; ++i) {
    const double x = static_cast<double>(input_vertices[2 * i]);
    const double y = static_cast<double>(input_vertices[2 * i + 1]);
    handles[i] = cdt.insert(CDT2DH::Point(x, y));
  }

  try {
    size_t offset = 0;
    for (auto len : subpath_lengths) {
      for (size_t i = 0; i < len; ++i) {
        const size_t cur = offset + i;
        const size_t next = offset + ((i + 1) % len);
        if (handles[cur] != handles[next]) {
          cdt.insert_constraint(handles[cur], handles[next]);
        }
      }
      offset += len;
    }
  } catch (const CDT2DH::Intersection_of_constraints_exception &) {
    return CDT2DResult("Self-intersecting or overlapping constraints detected");
  } catch (const std::exception &e) {
    return CDT2DResult(std::string("Error inserting constraints: ") + e.what());
  }

  if (cdt.dimension() < 2) {
    return CDT2DResult("Degenerate input - all points are collinear");
  }

  // Caller-supplied free interior points (e.g. distortion-driven Steiner points).  Inserted
  // unconstrained, before domain marking, so `mark_domain_in_triangulation` sets `is_in_domain()`
  // on the faces they create.  Callers must keep these inside the domain — an out-of-domain point
  // would surface as an unreferenced output vertex.
  for (size_t i = 0; i < interior_points.size() / 2; ++i) {
    cdt.insert(CDT2DH::Point(static_cast<double>(interior_points[2 * i]),
                             static_cast<double>(interior_points[2 * i + 1])));
  }

  // Flood-fills from the unbounded face across constraint edges, marking
  // alternating in/out by nesting depth.  Stored on each face via
  // `is_in_domain()` because we used `Delaunay_mesh_face_base_2`.
  CGAL::mark_domain_in_triangulation(cdt);

  if (refine) {
    try {
      CGAL::refine_Delaunay_mesh_2(
          cdt, CGAL::parameters::domain_is_initialized(true).criteria(
                   CDT2DH_Criteria(static_cast<double>(min_angle_bound),
                                   static_cast<double>(max_edge_len))));
    } catch (const std::exception &e) {
      return CDT2DResult(std::string("Error refining mesh: ") + e.what());
    }
  }

  std::vector<float> out_vertices;
  std::map<CDT2DH::Vertex_handle, uint32_t> vh_to_idx;

  for (auto vh = cdt.finite_vertices_begin(); vh != cdt.finite_vertices_end();
       ++vh) {
    vh_to_idx[vh] = static_cast<uint32_t>(out_vertices.size() / 2);
    out_vertices.push_back(static_cast<float>(vh->point().x()));
    out_vertices.push_back(static_cast<float>(vh->point().y()));
  }

  // Refinement adds Steiner points with no input correspondent, so we only
  // emit the mapping in the non-refining case (where it's well-defined).
  std::vector<int32_t> vertex_mapping;
  if (!refine) {
    vertex_mapping.resize(num_input_verts);
    for (size_t i = 0; i < num_input_verts; ++i) {
      vertex_mapping[i] = static_cast<int32_t>(vh_to_idx[handles[i]]);
    }
  }

  std::vector<uint32_t> out_indices;
  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end();
       ++fit) {
    if (fit->is_in_domain()) {
      out_indices.push_back(vh_to_idx[fit->vertex(0)]);
      out_indices.push_back(vh_to_idx[fit->vertex(1)]);
      out_indices.push_back(vh_to_idx[fit->vertex(2)]);
    }
  }

  if (out_indices.empty()) {
    return CDT2DResult("No interior triangles found - check subpath nesting "
                       "and constraint topology");
  }

  return CDT2DResult(std::move(out_vertices), std::move(out_indices),
                     std::move(vertex_mapping));
}

// ---------------------------------------------------------------------------
// 2D Boolean set operations on polygons (exact-arithmetic alternative to
// Clipper2).  Inputs are accepted in the same flat-coords + per-subpath-lengths
// format as the CDT-with-holes function.  Within each input, subpaths are
// combined under even-odd / XOR semantics (each subpath toggles inside/outside
// of overlapping regions), then the chosen boolean op is applied between
// subject and clip.  Output is flattened in the same format: outer boundaries
// emitted CCW followed by their holes CW, one polygon-with-holes after another.
// ---------------------------------------------------------------------------
typedef CGAL::Exact_predicates_exact_constructions_kernel EK;
typedef EK::Point_2 EK_Point_2;
typedef CGAL::Polygon_2<EK> EK_Polygon_2;
typedef CGAL::Polygon_with_holes_2<EK> EK_Polygon_with_holes_2;
typedef CGAL::Polygon_set_2<EK> EK_Polygon_set_2;

class PolygonBoolean2DResult {
private:
  std::vector<float> coords_;
  std::vector<uint32_t> path_lengths_;
  bool success_;
  std::string error_;

public:
  PolygonBoolean2DResult() : success_(false), error_("Not initialized") {}
  PolygonBoolean2DResult(std::vector<float> coords,
                         std::vector<uint32_t> path_lengths)
      : coords_(std::move(coords)), path_lengths_(std::move(path_lengths)),
        success_(true) {}
  PolygonBoolean2DResult(std::string error)
      : success_(false), error_(std::move(error)) {}

  bool success() const { return success_; }
  std::string getError() const { return error_; }
  std::vector<float> getCoords() const { return coords_; }
  std::vector<uint32_t> getPathLengths() const { return path_lengths_; }
};

static std::vector<Kernel::Point_2> to_points_2(const std::vector<float> &flat) {
  std::vector<Kernel::Point_2> points;
  points.reserve(flat.size() / 2);
  for (size_t i = 0; i + 1 < flat.size(); i += 2) {
    points.emplace_back(flat[i], flat[i + 1]);
  }
  return points;
}

// rings are emitted outer boundary first, then its holes, for each polygon
static PolygonBoolean2DResult
flattenMultipolygon(const CGAL::Multipolygon_with_holes_2<Kernel> &mp) {
  std::vector<float> coords;
  std::vector<uint32_t> lengths;
  auto push_ring = [&](const CGAL::Polygon_2<Kernel> &ring) {
    lengths.push_back(ring.size());
    for (const auto &p : ring) {
      coords.push_back(static_cast<float>(p.x()));
      coords.push_back(static_cast<float>(p.y()));
    }
  };
  for (const auto &pwh : mp.polygons_with_holes()) {
    push_ring(pwh.outer_boundary());
    for (const auto &hole : pwh.holes()) {
      push_ring(hole);
    }
  }
  return PolygonBoolean2DResult(std::move(coords), std::move(lengths));
}

template <class Input>
static PolygonBoolean2DResult
alphaWrap2DRun(const Input &input, const std::vector<Kernel::Point_2> &points,
               double relative_alpha, double relative_offset, bool manifold,
               const std::vector<float> &seeds) {
  if (points.empty()) {
    return PolygonBoolean2DResult("alpha_wrap_2d: input is empty");
  }
  const CGAL::Bbox_2 bbox = CGAL::bbox_2(points.begin(), points.end());
  const double diag = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                CGAL::square(bbox.ymax() - bbox.ymin()));
  const double alpha = diag * relative_alpha;
  const double offset = diag * relative_offset;
  if (!(alpha > 0.) || !(offset > 0.)) {
    return PolygonBoolean2DResult(
        "alpha_wrap_2d: alpha and offset must be > 0 and the input must have a "
        "non-zero extent");
  }
  const auto seed_points = to_points_2(seeds);
  CGAL::Multipolygon_with_holes_2<Kernel> out;
  try {
    CGAL::alpha_wrap_2(input, alpha, offset, out,
                       CGAL::parameters::do_enforce_manifoldness(manifold)
                           .seed_points(seed_points));
  } catch (const std::exception &e) {
    return PolygonBoolean2DResult(std::string("CGAL error in alpha_wrap_2d: ") +
                                  e.what());
  }
  return flattenMultipolygon(out);
}

// segments: flat [x0, y0, x1, y1, ...] per segment
PolygonBoolean2DResult alphaWrap2D(const std::vector<float> &segments,
                                   double relative_alpha, double relative_offset,
                                   bool manifold, const std::vector<float> &seeds) {
  const auto endpoints = to_points_2(segments);
  std::vector<Kernel::Segment_2> segs;
  segs.reserve(endpoints.size() / 2);
  for (size_t i = 0; i + 1 < endpoints.size(); i += 2) {
    segs.emplace_back(endpoints[i], endpoints[i + 1]);
  }
  return alphaWrap2DRun(segs, endpoints, relative_alpha, relative_offset, manifold, seeds);
}

PolygonBoolean2DResult alphaWrap2DPoints(const std::vector<float> &points,
                                         double relative_alpha, double relative_offset,
                                         bool manifold, const std::vector<float> &seeds) {
  const auto pts = to_points_2(points);
  return alphaWrap2DRun(pts, pts, relative_alpha, relative_offset, manifold, seeds);
}

static bool buildEKPolygonSet(EK_Polygon_set_2 &set,
                              const std::vector<float> &coords,
                              const std::vector<uint32_t> &subpath_lengths,
                              std::string &err) {
  size_t offset = 0;
  for (auto len : subpath_lengths) {
    if (len < 3) {
      offset += len;
      continue;
    }
    EK_Polygon_2 poly;
    for (size_t i = 0; i < len; ++i) {
      const double x = static_cast<double>(coords[2 * (offset + i)]);
      const double y = static_cast<double>(coords[2 * (offset + i) + 1]);
      poly.push_back(EK_Point_2(x, y));
    }
    offset += len;
    if (poly.size() < 3) {
      continue;
    }
    if (!poly.is_simple()) {
      err = "CGAL boolean: input subpath is self-intersecting (not simple)";
      return false;
    }
    if (poly.orientation() == CGAL::CLOCKWISE) {
      poly.reverse_orientation();
    }
    // EvenOdd / XOR build: nested subpaths cancel to form holes; non-overlapping
    // subpaths sum to a union.  Matches geoscript's nesting-based fill model.
    set.symmetric_difference(poly);
  }
  return true;
}

static void extractEKPolygonSet(const EK_Polygon_set_2 &set,
                                std::vector<float> &out_coords,
                                std::vector<uint32_t> &out_lengths) {
  std::vector<EK_Polygon_with_holes_2> result;
  set.polygons_with_holes(std::back_inserter(result));
  for (const auto &pwh : result) {
    if (pwh.is_unbounded()) {
      continue;
    }
    const auto &outer = pwh.outer_boundary();
    out_lengths.push_back(static_cast<uint32_t>(outer.size()));
    for (auto it = outer.vertices_begin(); it != outer.vertices_end(); ++it) {
      out_coords.push_back(static_cast<float>(CGAL::to_double(it->x())));
      out_coords.push_back(static_cast<float>(CGAL::to_double(it->y())));
    }
    for (auto h = pwh.holes_begin(); h != pwh.holes_end(); ++h) {
      out_lengths.push_back(static_cast<uint32_t>(h->size()));
      for (auto it = h->vertices_begin(); it != h->vertices_end(); ++it) {
        out_coords.push_back(static_cast<float>(CGAL::to_double(it->x())));
        out_coords.push_back(static_cast<float>(CGAL::to_double(it->y())));
      }
    }
  }
}

// op: 0=union, 1=intersect, 2=difference (subject - clip), 3=xor
PolygonBoolean2DResult
pathBoolean2D(const std::vector<float> &subject_coords,
              const std::vector<uint32_t> &subject_lengths,
              const std::vector<float> &clip_coords,
              const std::vector<uint32_t> &clip_lengths, int op) {
  try {
    EK_Polygon_set_2 subject_set;
    EK_Polygon_set_2 clip_set;
    std::string err;
    if (!buildEKPolygonSet(subject_set, subject_coords, subject_lengths, err)) {
      return PolygonBoolean2DResult(err);
    }
    if (!buildEKPolygonSet(clip_set, clip_coords, clip_lengths, err)) {
      return PolygonBoolean2DResult(err);
    }

    EK_Polygon_set_2 result_set(subject_set);
    switch (op) {
    case 0:
      result_set.join(clip_set);
      break;
    case 1:
      result_set.intersection(clip_set);
      break;
    case 2:
      result_set.difference(clip_set);
      break;
    case 3:
      result_set.symmetric_difference(clip_set);
      break;
    default:
      return PolygonBoolean2DResult("CGAL boolean: unknown op id " +
                                    std::to_string(op));
    }

    std::vector<float> out_coords;
    std::vector<uint32_t> out_lengths;
    extractEKPolygonSet(result_set, out_coords, out_lengths);
    return PolygonBoolean2DResult(std::move(out_coords), std::move(out_lengths));
  } catch (const std::exception &e) {
    return PolygonBoolean2DResult(std::string("CGAL boolean error: ") +
                                  e.what());
  }
}

EMSCRIPTEN_BINDINGS(my_module) {
  register_vector_custom<float>("vector<float>");
  register_vector_custom<uint32_t>("vector<uint32_t>");
  register_vector_custom<int32_t>("vector<int32_t>");

  emscripten::class_<PolyMesh>("PolyMesh")
      .constructor<>()
      .function("buildFromBuffers", &PolyMesh::buildFromBuffers,
                emscripten::allow_raw_pointers())
      .function("set_feature_edges", &PolyMesh::set_feature_edges)
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
                emscripten::allow_raw_pointers())
      .function("getLastError", &PolyMesh::getLastError,
                emscripten::allow_raw_pointers());

  emscripten::function("alphaWrapPointCloud", &alphaWrapPointCloud,
                       emscripten::allow_raw_pointers());

  // 2D Constrained Delaunay Triangulation
  emscripten::class_<CDT2DResult>("CDT2DResult")
      .constructor<>()
      .function("success", &CDT2DResult::success)
      .function("getError", &CDT2DResult::getError)
      .function("getVertices", &CDT2DResult::getVertices)
      .function("getIndices", &CDT2DResult::getIndices)
      .function("getVertexMapping", &CDT2DResult::getVertexMapping)
      .function("getVertexCount", &CDT2DResult::getVertexCount)
      .function("getTriangleCount", &CDT2DResult::getTriangleCount);

  emscripten::function("triangulatePolygon2D", &triangulatePolygon2D,
                       emscripten::allow_raw_pointers());
  emscripten::function("triangulatePolygon2DWithHoles",
                       &triangulatePolygon2DWithHoles,
                       emscripten::allow_raw_pointers());

  // 2D Boolean Set Operations
  emscripten::class_<PolygonBoolean2DResult>("PolygonBoolean2DResult")
      .constructor<>()
      .function("success", &PolygonBoolean2DResult::success)
      .function("getError", &PolygonBoolean2DResult::getError)
      .function("getCoords", &PolygonBoolean2DResult::getCoords)
      .function("getPathLengths", &PolygonBoolean2DResult::getPathLengths);

  emscripten::function("pathBoolean2D", &pathBoolean2D,
                       emscripten::allow_raw_pointers());
  emscripten::function("alphaWrap2D", &alphaWrap2D,
                       emscripten::allow_raw_pointers());
  emscripten::function("alphaWrap2DPoints", &alphaWrap2DPoints,
                       emscripten::allow_raw_pointers());
}
