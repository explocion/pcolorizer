#pragma once

// Just to provide function declarations for Clangd
// Including graph implementation headers before any headers in this library is
// still a requirement!!!
#include "undirected_graph.h"

#include "concepts.h"

#include <cmath>

#include <boost/range/adaptors.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/numeric.hpp>

namespace pcolorizer {

template <typename _Derived> class GraphColorizer {
public:
  using Derived = _Derived;

  Derived &derived() noexcept { return static_cast<Derived &>(*this); }

  const Derived &derived() const noexcept {
    return static_cast<const Derived &>(*this);
  }

  template <typename Graph, typename size_type = typename boost::graph_traits<
                                Graph>::vertices_size_type>
  size_type min_colors(const Graph &graph) {
    BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

    return this->derived().template min_colors<Graph, size_type>(graph);
  }

public:
  template <typename Graph, typename size_type = typename boost::graph_traits<
                                Graph>::vertices_size_type>
  static size_type clique_inf(const Graph &graph) {
    BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

    auto num_vertices = boost::num_vertices(graph);
    auto terms = boost::make_iterator_range(boost::vertices(graph)) |
                 boost::adaptors::transformed([&graph, num_vertices](auto v) {
                   auto d = boost::out_degree(v, graph);
                   return 1.0f / (num_vertices - d);
                 });
    return static_cast<size_type>(std::ceil(boost::accumulate(terms, 0.0f)));
  }

  template <typename Graph, typename size_type = typename boost::graph_traits<
                                Graph>::vertices_size_type>
  static size_type combination_sup(const Graph &graph) {
    BOOST_CONCEPT_ASSERT((pcolorizer::concepts::VertexColorableGraph<Graph>));

    auto num_edges = boost::num_edges(graph);
    return static_cast<size_type>(0.5f + std::sqrtf(0.25f + 2 * num_edges));
  }
};

} // namespace pcolorizer
