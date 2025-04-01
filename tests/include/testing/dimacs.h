#pragma once

#include "undirected_graph.h"

#include <boost/graph/graph_concepts.hpp>
#include <iostream>
#include <optional>

#include <boost/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>

namespace dimacs {

namespace phoenix = boost::phoenix;
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

template <typename Graph = pcolorizer::MutableUndirectedGraph<>>
std::optional<Graph> read_dimacs_coloring(std::istream &dimacs) {
  using vertex_size_type =
      typename boost::graph_traits<Graph>::vertices_size_type;
  using edge_size_type = typename boost::graph_traits<Graph>::edges_size_type;

  BOOST_CONCEPT_ASSERT((boost::concepts::MutableGraph<Graph>));
  BOOST_CONCEPT_ASSERT((boost::concepts::EdgeListGraph<Graph>));

  boost::spirit::istream_iterator begin(dimacs >> std::noskipws), end;

  vertex_size_type num_vertices = 0;
  auto num_vertices_ = phoenix::ref(num_vertices);
  qi::uint_parser<vertex_size_type> vertex_;

  edge_size_type num_edges = 0;
  auto num_edges_ = phoenix::ref(num_edges);
  qi::uint_parser<edge_size_type> edge_;

  auto comment = boost::proto::deep_copy(
      qi::lexeme[qi::lit('c') >> *(qi::char_ - qi::eol)]);
  auto skipper = ascii::space | comment;

  auto problem = qi::lit('p') >> qi::lit("edge") >>
                 vertex_[num_vertices_ = qi::_1] >> edge_[num_edges_ = qi::_1];

  if (!qi::phrase_parse(begin, end, problem, skipper)) {
    return std::nullopt;
  }

  Graph graph(num_vertices);
  auto add_edge_ = [&graph](vertex_size_type u, vertex_size_type v) {
    boost::add_edge(u - 1, v - 1, graph);
  };
  auto edge = boost::proto::deep_copy(
      (qi::lit('e') >> vertex_ >>
       vertex_)[phoenix::bind(add_edge_, qi::_1, qi::_2)]);

  if (!qi::phrase_parse(begin, end, *edge, skipper) ||
      boost::num_edges(graph) != num_edges / 2) {
    return std::nullopt;
  }

  return graph;
}

} // namespace dimacs
