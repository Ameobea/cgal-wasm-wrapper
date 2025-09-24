#define CGAL_EIGEN3_ENABLED

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

typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> SurfaceMesh;

// Helper function to get vector data pointer
template <typename T> intptr_t getVecDataPtr(std::vector<T> &vec) {
  return reinterpret_cast<intptr_t>(vec.data());
}

// Template function to register vector types for emscripten
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
  // SurfaceMesh triangulatedMesh;
public:
  SurfaceMesh mesh;
  Kernel::Point_3 sdf_p = Kernel::Point_3(0, 4, 0);
  PolyMesh() { std::cout << "created mesh" << std::endl; }

  // Create mesh from raw vertex positions and indices
  void buildFromBuffers(const std::vector<float> &vertices,
                        const std::vector<uint32_t> &indices) {
    // Clear existing mesh
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

    std::cout << "Building mesh from " << (vertices.size() / 3)
              << " vertices and " << (indices.size() / 3) << " faces"
              << std::endl;

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

      if (i0 >= vertex_indices.size() || i1 >= vertex_indices.size() ||
          i2 >= vertex_indices.size()) {
        std::cout << "Warning: Invalid index in face " << (i / 3)
                  << " (indices: " << i0 << ", " << i1 << ", " << i2
                  << ", max: " << (vertex_indices.size() - 1) << ")"
                  << std::endl;
        continue;
      }

      std::vector<SurfaceMesh::vertex_index> face_verts = {
          vertex_indices[i0], vertex_indices[i1], vertex_indices[i2]};

      mesh.add_face(face_verts);
    }

    std::cout << "Mesh built successfully with " << mesh.number_of_vertices()
              << " vertices and " << mesh.number_of_faces() << " faces"
              << std::endl;
  }

  // CGAL::SM_Vertex_index
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
  void catmull_smooth() {
    CGAL::Subdivision_method_3::CatmullClark_subdivision(mesh, 1);
  }
  void loop_smooth() { CGAL::Subdivision_method_3::Loop_subdivision(mesh, 1); }
  void dooSabin_smooth() {
    CGAL::Subdivision_method_3::DooSabin_subdivision(mesh, 1);
  }
  void sqrt_smooth() { CGAL::Subdivision_method_3::Sqrt3_subdivision(mesh, 1); }
  void decimate(double stop_ratio) {
    namespace SMS = CGAL::Surface_mesh_simplification;
    std::cout << "decimate stop ratio" << stop_ratio << std::endl;
    SMS::Count_ratio_stop_predicate<SurfaceMesh> stop(stop_ratio);
    int r = SMS::edge_collapse(mesh, stop);
    std::cout << "\nFinished!\n"
              << r << " edges removed.\n"
              << mesh.number_of_edges() << " final edges.\n";
  }

  emscripten::val getIndices() {
    std::vector<int> indices;
    SurfaceMesh triangulatedMesh = mesh;
    if (!this->isTriangulated(mesh)) {
      std::cout << "non-triangular mesh, triangulating..." << std::endl;
      this->triangulate(triangulatedMesh); // triangulate first
    }

    for (auto faceIt = triangulatedMesh.faces_begin();
         faceIt != triangulatedMesh.faces_end(); ++faceIt) {
      auto halfedgeIt = triangulatedMesh.halfedge(*faceIt);
      auto halfedgeIt2 = triangulatedMesh.next(halfedgeIt);
      auto halfedgeIt3 = triangulatedMesh.next(halfedgeIt2);
      auto vertexIt = triangulatedMesh.target(halfedgeIt);
      auto vertexIt2 = triangulatedMesh.target(halfedgeIt2);
      auto vertexIt3 = triangulatedMesh.target(halfedgeIt3);
      indices.push_back(vertexIt.idx());
      indices.push_back(vertexIt2.idx());
      indices.push_back(vertexIt3.idx());
    }
    triangulatedMesh.clear();
    triangulatedMesh.collect_garbage();
    return emscripten::val(
        emscripten::typed_memory_view(indices.size(), indices.data()));
  }

  bool isTriangulated(const SurfaceMesh &mesh) {
    for (const auto &f : mesh.faces()) {
      if (mesh.degree(f) != 3) {
        return false; // Face does not have 3 vertices, not triangulated
      }
    }
    return true;
  }

  emscripten::val getVertices() {
    std::vector<double> vertices;
    for (auto vertexIt = mesh.vertices_begin(); vertexIt != mesh.vertices_end();
         ++vertexIt) {
      auto point = mesh.point(*vertexIt);
      vertices.push_back(point.x());
      vertices.push_back(point.y());
      vertices.push_back(point.z());
    }
    return emscripten::val(
        emscripten::typed_memory_view(vertices.size(), vertices.data()));
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
    std::cout << "before sdf values" << std::endl;
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
      std::cout << "using default parameters" << std::endl;
      number_of_segments = CGAL::segmentation_from_sdf_values(
          mesh, sdf_property_map, segment_property_map);
    } else {
      std::cout << "using custom parameters" << std::endl;
      number_of_segments = CGAL::segmentation_from_sdf_values(
          mesh, sdf_property_map, segment_property_map, number_of_clusters,
          smoothing_lambda);
    }
    std::cout << "sdf values" << std::endl;
    std::cout << "Number of segments: " << number_of_segments << std::endl;
    // get face descriptor os surfacemesh
    std::vector<int> segments;
    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end();
         ++faceIt) {
      std::cout << segment_property_map[*faceIt] << " ";
      segments.push_back(segment_property_map[*faceIt]);
    }
    std::cout << std::endl;
    return emscripten::val(
        emscripten::typed_memory_view(segments.size(), segments.data()));
  }

  PolyMesh alphaWrap(double relative_alpha, double relative_offset) {
    namespace PMP = CGAL::Polygon_mesh_processing;

    if (mesh.is_empty()) {
      std::cout << "Warning: Cannot perform alpha wrap on empty mesh"
                << std::endl;
      return PolyMesh();
    }

    CGAL::Bbox_3 bbox = PMP::bbox(mesh);
    const double diag_length =
        std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                  CGAL::square(bbox.ymax() - bbox.ymin()) +
                  CGAL::square(bbox.zmax() - bbox.zmin()));

    const double alpha = diag_length / relative_alpha;
    const double offset = diag_length / relative_offset;

    std::cout << "Alpha wrap parameters - alpha: " << alpha
              << ", offset: " << offset << std::endl;

    PolyMesh result;

    try {
      CGAL::alpha_wrap_3(mesh, alpha, offset, result.mesh);
      std::cout << "Alpha wrap completed. Result: "
                << result.mesh.number_of_vertices() << " vertices, "
                << result.mesh.number_of_faces() << " faces" << std::endl;
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
    std::cout << "Error: Empty point cloud" << std::endl;
    return PolyMesh();
  }

  std::cout << "Processing point cloud with " << points.size() << " points"
            << std::endl;

  CGAL::Bbox_3 bbox = CGAL::bbox_3(std::cbegin(points), std::cend(points));
  const double diag_length = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                       CGAL::square(bbox.ymax() - bbox.ymin()) +
                                       CGAL::square(bbox.zmax() - bbox.zmin()));
  const double alpha = diag_length / relative_alpha;
  const double offset = diag_length / relative_offset;

  std::cout << "Alpha wrap parameters - alpha: " << alpha
            << ", offset: " << offset << std::endl;

  PolyMesh result;

  try {
    CGAL::alpha_wrap_3(points, alpha, offset, result.mesh);
    std::cout << "Point cloud alpha wrap completed. Result: "
              << result.mesh.number_of_vertices() << " vertices, "
              << result.mesh.number_of_faces() << " faces" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Error during point cloud alpha wrapping: " << e.what()
              << std::endl;
  }

  return result;
}

EMSCRIPTEN_BINDINGS(my_module) {
  // Register vector types
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
