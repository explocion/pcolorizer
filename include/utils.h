#pragma once

#include "concepts.h"

#include <vector>

// Just to provide function declarations for Clangd
// Including graph implementation headers before any headers in this library is
// still a requirement!!!
#include <boost/graph/adjacency_list.hpp>

#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/range/algorithm.hpp>

namespace pcolorizer {

namespace connected {

template <typename Graph> bool is_connected_graph(const Graph &graph) {
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  std::vector<typename boost::graph_traits<Graph>::vertices_size_type>
      components((boost::num_vertices(graph)));
  auto num_components = boost::connected_components(graph, components.begin());

  return num_components <= 1;
}

template <typename Graph> bool is_complete_graph(const Graph &graph) {
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  auto n = boost::num_vertices(graph);
  return n * (n - 1) == 2 * boost::num_edges(graph);
}

template <typename Graph> bool is_cycle_graph(const Graph &graph) {
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  auto vertices = boost::make_iterator_range(boost::vertices(graph));
  return boost::count_if(vertices, [&graph](auto v) {
           return boost::out_degree(v, graph) != 2;
         }) == 0;
}

} // namespace connected

template <typename Graph>
typename Graph::degree_size_type max_degree(const Graph &graph) {
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  auto vertices = boost::make_iterator_range(boost::vertices(graph));
  auto it = boost::max_element(vertices, [&graph](auto u, auto v) {
    return boost::out_degree(u, graph) < boost::out_degree(v, graph);
  });
  return (it != vertices.end()) ? boost::out_degree(*it, graph) : 0;
}

} // namespace pcolorizer