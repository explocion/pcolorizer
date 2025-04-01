#pragma once

#include "concepts.h"
#include "undirected_graph.h"

namespace generators {

template <typename Graph = pcolorizer::MutableUndirectedGraph<>>
Graph make_complete_graph(std::size_t n) {
  BOOST_CONCEPT_ASSERT((boost::concepts::MutableGraph<Graph>));
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  Graph graph(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      boost::add_edge(i, j, graph);
    }
  }
  return graph;
}

template <typename Graph = pcolorizer::MutableUndirectedGraph<>>
Graph make_cycle_graph(std::size_t n) {
  BOOST_CONCEPT_ASSERT((boost::concepts::MutableGraph<Graph>));
  BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

  Graph graph(n);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    boost::add_edge(i, i + 1, graph);
  }
  if (n >= 3) {
    boost::add_edge(n - 1, 0, graph);
  }
  return graph;
}

} // namespace generators