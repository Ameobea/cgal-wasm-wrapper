#define CGAL_EIGEN3_ENABLED
#define CGAL_ALWAYS_ROUND_TO_NEAREST
#define __wasm__
#define __SSE2__
#define __SSE4_1__

#include <CGAL/Dimension.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
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

      mesh.add_face(face_verts);
    }
  }

  int addVertex(double x, double y, double z) {
    auto v0 = mesh.add_vertex(Kernel::Point_3(x, y, z));
    return v0.idx();
  }

  int addFace(emscripten::val vertices) {
    std::vector<SurfaceMesh::vertex_index> faceVerts;
    for (int i = 0; i < vertices["length"].as<int>(); i++) {
      faceVerts.push_back(mesh.vertices().begin()[vertices[i].as<int>()]);
    }
    auto fc = mesh.add_face(faceVerts);
    return fc.idx();
  }

  void triangulate(SurfaceMesh &targetMesh) {
    CGAL::Polygon_mesh_processing::triangulate_faces(targetMesh);
  }

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

  void decimate(double stop_ratio) {
    namespace SMS = CGAL::Surface_mesh_simplification;
    SMS::Count_ratio_stop_predicate<SurfaceMesh> stop(stop_ratio);
    int r = SMS::edge_collapse(mesh, stop);
  }

  std::vector<uint32_t> getIndices() {
    std::vector<uint32_t> indices;
    SurfaceMesh triangulatedMesh = mesh;
    if (!this->isTriangulated(mesh)) {
      this->triangulate(triangulatedMesh);
    }

    for (auto faceIt = triangulatedMesh.faces_begin();
         faceIt != triangulatedMesh.faces_end(); ++faceIt) {
      auto halfedgeIt = triangulatedMesh.halfedge(*faceIt);
      auto halfedgeIt2 = triangulatedMesh.next(halfedgeIt);
      auto halfedgeIt3 = triangulatedMesh.next(halfedgeIt2);
      auto vertexIt = triangulatedMesh.target(halfedgeIt);
      auto vertexIt2 = triangulatedMesh.target(halfedgeIt2);
      auto vertexIt3 = triangulatedMesh.target(halfedgeIt3);
      indices.push_back(static_cast<uint32_t>(vertexIt.idx()));
      indices.push_back(static_cast<uint32_t>(vertexIt2.idx()));
      indices.push_back(static_cast<uint32_t>(vertexIt3.idx()));
    }
    triangulatedMesh.clear();
    triangulatedMesh.collect_garbage();
    return indices;
  }

  bool isTriangulated(const SurfaceMesh &mesh) {
    for (const auto &f : mesh.faces()) {
      if (mesh.degree(f) != 3) {
        return false;
      }
    }
    return true;
  }

  std::vector<float> getVertices() {
    std::vector<float> vertices;
    SurfaceMesh triangulatedMesh = mesh;
    if (!this->isTriangulated(mesh)) {
      this->triangulate(triangulatedMesh);
    }

    for (auto vertexIt = mesh.vertices_begin(); vertexIt != mesh.vertices_end();
         ++vertexIt) {
      auto point = mesh.point(*vertexIt);
      vertices.push_back(static_cast<float>(point.x()));
      vertices.push_back(static_cast<float>(point.y()));
      vertices.push_back(static_cast<float>(point.z()));
    }

    triangulatedMesh.clear();
    triangulatedMesh.collect_garbage();

    return vertices;
  }

  emscripten::val segment(int n_clusters, double sm_lambda) {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

    typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor
        vertex_descriptor;
    typedef boost::graph_traits<SurfaceMesh>::face_descriptor face_descriptor;

    // create a property-map
    typedef SurfaceMesh::Property_map<face_descriptor, double> Facet_double_map;
    Facet_double_map sdf_property_map;
    sdf_property_map =
        mesh.add_property_map<face_descriptor, double>("f:sdf").first;
    // compute SDF values
    // We can't use default parameters for number of rays, and cone angle
    // and the postprocessing
    CGAL::sdf_values(mesh, sdf_property_map, 2.0 / 3.0 * CGAL_PI, 25, true);
    // create a property-map for segment-ids
    typedef SurfaceMesh::Property_map<face_descriptor, std::size_t>
        Facet_int_map;
    Facet_int_map segment_property_map =
        mesh.add_property_map<face_descriptor, std::size_t>("f:sid").first;
    ;

    const std::size_t number_of_clusters =
        n_clusters; // use 4 clusters in soft clustering
    const double smoothing_lambda =
        sm_lambda; // importance of surface features, suggested to be in-between
                   // [0,1]

    std::size_t number_of_segments;
    if (n_clusters == 0 && sm_lambda == 0) {
      number_of_segments = CGAL::segmentation_from_sdf_values(
          mesh, sdf_property_map, segment_property_map);
    } else {
      number_of_segments = CGAL::segmentation_from_sdf_values(
          mesh, sdf_property_map, segment_property_map, number_of_clusters,
          smoothing_lambda);
    }
    // get face descriptor os surfacemesh
    std::vector<int> segments;
    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end();
         ++faceIt) {
      segments.push_back(segment_property_map[*faceIt]);
    }
    return emscripten::val(
        emscripten::typed_memory_view(segments.size(), segments.data()));
  }

  PolyMesh alphaWrap(double relative_alpha, double relative_offset) {
    namespace PMP = CGAL::Polygon_mesh_processing;

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
      .function("addVertex", &PolyMesh::addVertex,
                emscripten::allow_raw_pointers())
      .function("addFace", &PolyMesh::addFace, emscripten::allow_raw_pointers())
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
      .function("segment", &PolyMesh::segment, emscripten::allow_raw_pointers())
      .function("decimate", &PolyMesh::decimate,
                emscripten::allow_raw_pointers())
      .function("alphaWrap", &PolyMesh::alphaWrap,
                emscripten::allow_raw_pointers());

  emscripten::function("alphaWrapPointCloud", &alphaWrapPointCloud,
                       emscripten::allow_raw_pointers());
}
