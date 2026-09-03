#!/usr/bin/env python3
"""Patches CGAL's remesh_planar_patches.h so that, when the decimated polygon soup fails
is_polygon_soup_a_polygon_mesh(), the patches whose triangulations are inconsistent with their
neighbors are copied verbatim from the input and their neighbors remeshed again, instead of the
whole result being empty. Nothing extra runs when the soup is valid.
Usage: patch-remesh-planar-patches.py <path/to/remesh_planar_patches.h>  (idempotent)"""
import sys

path = sys.argv[1]
s = open(path).read()
if 'inconsistent_patches' in s:
    print('already patched')
    sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

rep('#include <boost/container/small_vector.hpp>\n',
    '#include <boost/container/small_vector.hpp>\n#include <set>\n')

rep('''template <typename Kernel,
          typename TriangleMesh,
          typename VertexCornerIdMap,
          typename EdgeIsConstrainedMap,
          typename FaceCCIdMap,
          typename VertexPointMap,
          typename IndexTracking>
bool decimate_impl(const TriangleMesh& tm,
''', '''// Patches are triangulated independently in their own projections, so a patch with a nearly
// degenerate boundary (e.g. a sliver notch at the scale of floating point noise) can claim a
// triangle its neighbor also claims, or pinch a corner. Returns the patches whose polygons
// duplicate a directed edge, repeat a vertex, or form more than one fan around a corner, ignoring
// patches copied verbatim from the input, since those are consistent by construction.
// Linear in the number of polygon corners: one slot per corner occurrence, slots bucketed by
// corner (counting sort), fans tracked by a union-find over slots joined across manifold edges.
template <class PolygonRange>
std::set<std::size_t> inconsistent_patches(const PolygonRange& faces,
                                           const std::vector<std::size_t>& face_cc,
                                           const boost::dynamic_bitset<>& copied_verbatim,
                                           std::size_t nb_corners)
{
  std::set<std::size_t> bad;
  auto flag = [&bad, &copied_verbatim](std::size_t cc_id)
  {
    if (!copied_verbatim.test(cc_id)) bad.insert(cc_id);
  };

  std::size_t nb_slots = 0;
  for (const auto& poly : faces)
    nb_slots += poly.size();

  // the directed edge of slot i goes from corner[i] to corner[next[i]]
  std::vector<std::size_t> corner(nb_slots), cc(nb_slots), next(nb_slots), offsets(nb_corners+1, 0), tmp;
  std::size_t s = 0;
  for (std::size_t f=0; f<faces.size(); ++f)
  {
    const auto& poly = faces[f];
    tmp.assign(poly.begin(), poly.end());
    std::sort(tmp.begin(), tmp.end());
    if (std::adjacent_find(tmp.begin(), tmp.end())!=tmp.end())
      flag(face_cc[f]);
    std::size_t first = s;
    for (std::size_t k=0; k<poly.size(); ++k, ++s)
    {
      corner[s] = poly[k];
      cc[s] = face_cc[f];
      next[s] = k+1<poly.size() ? s+1 : first;
      ++offsets[poly[k]+1];
    }
  }
  for (std::size_t c=0; c<nb_corners; ++c)
    offsets[c+1] += offsets[c];

  // bucket[offsets[c]..offsets[c+1]) holds the slots of corner c, sorted by edge target
  std::vector<std::size_t> bucket(nb_slots), pos(offsets.begin(), offsets.end()-1);
  for (std::size_t i=0; i<nb_slots; ++i)
    bucket[pos[corner[i]]++] = i;
  auto target = [&](std::size_t i) { return corner[next[i]]; };
  for (std::size_t c=0; c<nb_corners; ++c)
    std::sort(bucket.begin()+offsets[c], bucket.begin()+offsets[c+1],
              [&](std::size_t i, std::size_t j) { return target(i) < target(j); });

  std::vector<std::size_t> parent(nb_slots);
  for (std::size_t i=0; i<nb_slots; ++i) parent[i]=i;
  auto find = [&parent](std::size_t x) { while (parent[x]!=x) x = parent[x] = parent[parent[x]]; return x; };
  auto unite = [&](std::size_t x, std::size_t y) { parent[find(x)] = find(y); };

  for (std::size_t a=0; a<nb_corners; ++a)
  {
    for (std::size_t p=offsets[a]; p<offsets[a+1]; ++p)
    {
      std::size_t i = bucket[p], b = target(i);
      if (p+1<offsets[a+1] && target(bucket[p+1])==b)
      {
        flag(cc[i]);
        flag(cc[bucket[p+1]]);
        continue;
      }
      if (p>offsets[a] && target(bucket[p-1])==b) continue;
      if (b<a) continue; // each undirected edge is paired once, from its smaller corner
      auto lo = std::lower_bound(bucket.begin()+offsets[b], bucket.begin()+offsets[b+1], a,
                                 [&](std::size_t j, std::size_t v) { return target(j) < v; });
      if (lo==bucket.begin()+offsets[b+1] || target(*lo)!=a) continue; // border edge
      if (lo+1!=bucket.begin()+offsets[b+1] && target(*(lo+1))==a) continue; // duplicated, flagged from b
      std::size_t j = *lo;
      unite(i, next[j]);
      unite(next[i], j);
    }
  }

  for (std::size_t c=0; c<nb_corners; ++c)
  {
    std::size_t r = find(bucket[offsets[c]]);
    for (std::size_t p=offsets[c]+1; p<offsets[c+1]; ++p)
    {
      if (find(bucket[p])!=r)
      {
        for (std::size_t q=offsets[c]; q<offsets[c+1]; ++q) flag(cc[bucket[q]]);
        break;
      }
    }
  }
  return bad;
}

template <typename Kernel,
          typename TriangleMesh,
          typename VertexCornerIdMap,
          typename EdgeIsConstrainedMap,
          typename FaceCCIdMap,
          typename VertexPointMap,
          typename IndexTracking>
bool decimate_impl(const TriangleMesh& tm,
''')

rep('''  bool all_patches_successfully_remeshed = true;
  do
  {
    std::vector< std::vector<Id_pair> > face_boundaries(nb_corners_and_nb_cc.second);
    std::vector<bool> face_boundaries_valid(nb_corners_and_nb_cc.second, true);
''', '''  bool all_patches_successfully_remeshed = true;
  std::vector<bool> face_boundaries_valid(nb_corners_and_nb_cc.second, true);
  boost::dynamic_bitset<> copied_verbatim(nb_corners_and_nb_cc.second);

  // replaces the triangles of a patch by its input faces, all of its vertices becoming corners,
  // and schedules its neighbors to be remeshed again against the new corners
  auto copy_patch_verbatim = [&](std::size_t cc_id)
  {
    all_patches_successfully_remeshed = false;
    copied_verbatim.set(cc_id);
    std::vector< boost::container::small_vector<std::size_t,3> >& cc_faces = faces_per_cc[cc_id];
    cc_faces.clear();
    CGAL::Face_filtered_graph<TriangleMesh> ffg(tm, static_cast<PID>(cc_id), face_cc_ids);
    std::vector<vertex_descriptor> new_corners;
    for (vertex_descriptor v : vertices(ffg))
    {
      std::size_t i = get(vertex_corner_id, v);
      if ( !is_corner_id(i) )
      {
        i = nb_corners_and_nb_cc.first++;
        put(vertex_corner_id, v, i);
        corners.push_back(get(vpm, v));
        new_corners.push_back(v);
      }
    }
    for (face_descriptor f : faces(ffg))
    {
      halfedge_descriptor h = halfedge(f, tm);
      cc_faces.push_back({ get(vertex_corner_id, source(h,tm)),
                           get(vertex_corner_id, target(h,tm)),
                           get(vertex_corner_id, target(next(h,tm), tm)) });
    }
    // reset flag for neighbor connected components only if interface has changed
    for (vertex_descriptor v : new_corners)
    {
      for (halfedge_descriptor h : halfedges_around_target(halfedge(v, tm), tm))
      {
        if (!is_border(h, tm))
        {
          std::size_t other_cc_id = get(face_cc_ids, face(h, tm));
          cc_to_handle.set(other_cc_id, 1);
          face_boundaries_valid[ other_cc_id ]=false;
        }
      }
    }
    cc_to_handle.set(cc_id, 0);
  };

  bool valid = false;
  std::vector<std::size_t> cc_offsets; // out_faces[cc_offsets[cc]..cc_offsets[cc+1]) come from patch cc
  while (true)
  {
  do
  {
    std::vector< std::vector<Id_pair> > face_boundaries(nb_corners_and_nb_cc.second);
    std::fill(face_boundaries_valid.begin(), face_boundaries_valid.end(), true);
''')

rep('''      std::vector< Id_pair >& csts = face_boundaries[cc_id];

      if (!face_boundaries_valid[cc_id]) continue;
''', '''      std::vector< Id_pair >& csts = face_boundaries[cc_id];

      if (!face_boundaries_valid[cc_id]) continue;
      copied_verbatim.reset(cc_id);
''')

rep('''          ++fail_case_id;
#endif
          all_patches_successfully_remeshed = false;
          // make all vertices of the patch a corner
          CGAL::Face_filtered_graph<TriangleMesh> ffg(tm, static_cast<PID>(cc_id), face_cc_ids);
          std::vector<vertex_descriptor> new_corners;
          for (vertex_descriptor v : vertices(ffg))
          {
            std::size_t i = get(vertex_corner_id, v);
            if ( !is_corner_id(i) )
            {
              i = nb_corners_and_nb_cc.first++;
              put(vertex_corner_id, v, i);
              corners.push_back(get(vpm, v));
              new_corners.push_back(v);
            }
          }
          // add all the faces of the current patch
          for (face_descriptor f : faces(ffg))
          {
            halfedge_descriptor h = halfedge(f, tm);
            cc_faces.push_back({ get(vertex_corner_id, source(h,tm)),
                                 get(vertex_corner_id, target(h,tm)),
                                  get(vertex_corner_id, target(next(h,tm), tm)) });
          }
          // reset flag for neighbor connected components only if interface has changed
          for (vertex_descriptor v : new_corners)
          {
            for (halfedge_descriptor h : halfedges_around_target(halfedge(v, tm), tm))
            {
              if (!is_border(h, tm))
              {
                std::size_t other_cc_id = get(face_cc_ids, face(h, tm));
                cc_to_handle.set(other_cc_id, 1);
                face_boundaries_valid[ other_cc_id ]=false;
              }
            }
          }
          cc_to_handle.set(cc_id, 0);
        }
      }
    }
  }
  while(cc_to_handle.any());
''', '''          ++fail_case_id;
#endif
          copy_patch_verbatim(cc_id);
        }
      }
    }
  }
  while(cc_to_handle.any());
''')

rep('''  while(cc_to_handle.any());

  std::size_t cc_id=0;
  for (const std::vector<boost::container::small_vector<std::size_t,3>>& cc_trs : faces_per_cc)
  {
    out_faces.insert(out_faces.end(), cc_trs.begin(), cc_trs.end());
    f_id_tracker.register_faces_of_cc(cc_trs.size(), cc_id++);
  }

  return all_patches_successfully_remeshed;
}
''', '''  while(cc_to_handle.any());

  out_faces.clear();
  cc_offsets.assign(1, 0);
  cc_offsets.reserve(nb_corners_and_nb_cc.second+1);
  f_id_tracker.clear();
  std::size_t cc_id=0;
  for (const std::vector<boost::container::small_vector<std::size_t,3>>& cc_trs : faces_per_cc)
  {
    out_faces.insert(out_faces.end(), cc_trs.begin(), cc_trs.end());
    cc_offsets.push_back(out_faces.size());
    f_id_tracker.register_faces_of_cc(cc_trs.size(), cc_id++);
  }
  // released before validating so that the peak memory of the valid case is unchanged;
  // rebuilt from the flat soup only when a repair is needed
  std::vector< std::vector< boost::container::small_vector<std::size_t,3> > >().swap(faces_per_cc);

  valid = is_polygon_soup_a_polygon_mesh(out_faces);
  if (valid) break;
  std::vector<std::size_t> face_cc(out_faces.size());
  faces_per_cc.resize(nb_corners_and_nb_cc.second);
  for (std::size_t c=0; c<nb_corners_and_nb_cc.second; ++c)
  {
    for (std::size_t f=cc_offsets[c]; f<cc_offsets[c+1]; ++f)
    {
      face_cc[f] = c;
      faces_per_cc[c].push_back(out_faces[f]);
    }
  }
  std::set<std::size_t> bad = inconsistent_patches(out_faces, face_cc, copied_verbatim, nb_corners_and_nb_cc.first);
  if (bad.empty()) break; // failure cannot be attributed to a patch
  for (std::size_t bad_cc : bad)
    copy_patch_verbatim(bad_cc);
  }

  if (soup_valid) *soup_valid = valid;
  return all_patches_successfully_remeshed;
}
''')

# inner decimate_impl reports soup validity to its caller
rep('''                   IndexTracking& f_id_tracker,
                   std::vector< typename Kernel::Vector_3 >& face_normals)
{
  typedef typename boost::graph_traits<TriangleMesh> graph_traits;
  typedef typename graph_traits::halfedge_descriptor halfedge_descriptor;
''', '''                   IndexTracking& f_id_tracker,
                   std::vector< typename Kernel::Vector_3 >& face_normals,
                   bool* soup_valid = nullptr)
{
  typedef typename boost::graph_traits<TriangleMesh> graph_traits;
  typedef typename graph_traits::halfedge_descriptor halfedge_descriptor;
''')

rep('''  void register_faces_of_cc(std::size_t nb_faces, std::size_t i)
  {
    face_ids.insert(face_ids.end(), nb_faces, i);
  }
''', '''  void register_faces_of_cc(std::size_t nb_faces, std::size_t i)
  {
    face_ids.insert(face_ids.end(), nb_faces, i);
  }

  void clear() { face_ids.clear(); }
''')

rep('''  void register_faces_of_cc(std::size_t /*nb_triangles*/, std::size_t /*in_patch_id*/) {}
''', '''  void register_faces_of_cc(std::size_t /*nb_triangles*/, std::size_t /*in_patch_id*/) {}
  void clear() {}
''')

# the result variable is named remeshing_failed (5.6) or decimate_success (6.x)
import re
pat = r'(  std::vector< boost::container::small_vector<std::size_t,3> > faces;\n)(  bool \w+ = decimate_impl<Kernel>\(tm_in,\n)'
assert len(re.findall(pat, s)) == 1
s = re.sub(pat, r'\1  bool soup_valid = true;\n\2', s)

rep('''                                                f_id_tracker,
                                                face_normals);

  if (!is_polygon_soup_a_polygon_mesh(faces))
  {
''', '''                                                f_id_tracker,
                                                face_normals,
                                                &soup_valid);

  if (!soup_valid)
  {
''')
open(path, 'w').write(s)
print('patched', path)
