#pragma once

// Just to provide function declarations for Clangd
// Including graph implementation headers before any headers in this library is
// still a requirement!!!
#include "undirected_graph.h"

#include "colorizer.h"
#include "utils.h"

#include <z3++.h>

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

    auto lower = GraphColorizer<SmtColorizer>::clique_inf(graph);
    auto upper = std::min(GraphColorizer<SmtColorizer>::combination_sup(graph),
                          static_cast<size_type>(max_degree(graph)) + 1);
    if (lower == upper) {
      return lower;
    }

    auto indices = boost::get(boost::vertex_index, graph);

    z3::set_param("parallel.enable", true);
    z3::context ctx;

    auto chi = ctx.int_const("chi");

    z3::expr_vector colors(ctx);
    colors.resize(num_vertices);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
      auto id = indices[v];
      auto color = ctx.int_const(std::to_string(id).c_str());
      colors.set(id, color);
    }

    auto bounds = ctx.bool_val(true);
    for (auto color : colors) {
      bounds = bounds && (color >= 0 && color < chi);
    }

    auto edge_distinctions = ctx.bool_val(true);
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
      auto u = indices[boost::source(e, graph)];
      auto v = indices[boost::target(e, graph)];
      edge_distinctions = edge_distinctions && (colors[u] != colors[v]);
    }

    auto assertions = bounds && edge_distinctions;

    z3::params optimizer_parameters(ctx);
    optimizer_parameters.set("timeout", 10000u);

    z3::optimize optimizer(ctx);
    optimizer.set(optimizer_parameters);
    optimizer.set_initial_value(chi, static_cast<int>(upper));
    optimizer.add(chi >= static_cast<int>(lower) &&
                  chi <= static_cast<int>(upper));
    optimizer.add(assertions);
    optimizer.minimize(chi);

    z3::expr_vector assumptions(ctx);
    assumptions.push_back(
        z3::implies(chi == static_cast<int>(upper), assertions));

    switch (optimizer.check(assumptions)) {
    case z3::sat: {
      auto model = optimizer.get_model();
      return static_cast<size_type>(model.eval(chi).get_numeral_int());
    }
    default: {
      auto model = optimizer.get_model();
      auto evaluated = model.eval(chi);
      try {
        return static_cast<size_type>(evaluated.get_numeral_int());
      } catch (...) {
        return upper;
      }
    }
    }
  }
};

} // namespace pcolorizer