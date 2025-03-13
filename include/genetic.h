#pragma once

// Just to provide function declarations for Clangd
// Including graph implementation headers before any headers in this library is
// still a requirement!!!
#include "undirected_graph.h"

#include "colorizer.h"
#include "utils.h"

namespace pcolorizer {

class GeneticColorizer : public GraphColorizer<GeneticColorizer> {
public:
  template <typename Graph, typename size_type = typename boost::graph_traits<
                                Graph>::vertices_size_type>
  size_type min_colors(const Graph &graph) {
    auto num_vertices = boost::num_vertices(graph);

    if (pcolorizer::connected::is_complete_graph(graph)) {
      return num_vertices;
    } else if (pcolorizer::connected::is_cycle_graph(graph)) {
      return (num_vertices % 2) + 2;
    }

    auto lower = GraphColorizer<GeneticColorizer>::clique_inf(graph);
    auto upper =
        std::min(GraphColorizer<GeneticColorizer>::combination_sup(graph),
                 static_cast<size_type>(max_degree(graph)) + 1);
    if (lower == upper) {
      return lower;
    }
  }
};

} // namespace pcolorizer