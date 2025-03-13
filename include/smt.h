#pragma once

#include "colorizer.h"
#include "utils.h"

#include <z3++.h>

// Just to provide function declarations for Clangd
// Including graph implementation headers before any headers in this library is
// still a requirement!!!
#include <boost/graph/adjacency_list.hpp>

#include <boost/range/adaptors.hpp>
#include <boost/range/iterator_range.hpp>

namespace pcolorizer {

class SmtColorizer : public GraphColorizer<SmtColorizer> {
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

    auto lower_bound = GraphColorizer<SmtColorizer>::clique_inf(graph);
    auto upper_bound =
        std::min(GraphColorizer<SmtColorizer>::combination_sup(graph),
                 static_cast<size_type>(max_degree(graph)));

    if (lower_bound == upper_bound) {
      return lower_bound;
    }

    z3::context ctx;

    auto vertex_indices = boost::get(boost::vertex_index, graph);

    z3::expr_vector vertex_colors(ctx);
    vertex_colors.resize(num_vertices);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
      auto id = vertex_indices[v];
      auto color = ctx.int_const(std::to_string(id).c_str());
      vertex_colors.set(id, color);
    }

    z3::expr edge_distinctions = ctx.bool_val(true);
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
      auto u = vertex_indices[boost::source(e, graph)];
      auto v = vertex_indices[boost::target(e, graph)];
      edge_distinctions =
          edge_distinctions && (vertex_colors[u] != vertex_colors[v]);
    }

    auto chi = ctx.int_const("chi");

    z3::optimize optimizer(ctx);
    optimizer.add(chi >= static_cast<int>(lower_bound) &&
                  chi <= static_cast<int>(upper_bound));
    optimizer.add(edge_distinctions);
    for (auto color : vertex_colors) {
      optimizer.add(color >= 0 && color < chi);
    }
    optimizer.minimize(chi);

    if (optimizer.check() == z3::sat) {
      auto model = optimizer.get_model();
      return static_cast<size_type>(model.eval(chi).get_numeral_int());
    } else {
      return upper_bound;
    }
  }
};

} // namespace pcolorizer