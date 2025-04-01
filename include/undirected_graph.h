#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>

namespace pcolorizer {

template <typename OutEdgeListS = boost::vecS,
          typename VertexListS = boost::vecS,
          typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property,
          typename GraphProperty = boost::no_property,
          typename EdgeListS = boost::listS>
using MutableUndirectedGraph =
    boost::adjacency_list<OutEdgeListS, VertexListS, boost::undirectedS,
                          VertexProperty, EdgeProperty, GraphProperty,
                          EdgeListS>;

template <typename Directed = boost::directedS,
          typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property,
          typename GraphProperty = boost::no_property,
          typename Vertex = std::size_t, typename EdgeIndex = Vertex>
using UndirectedGraph =
    boost::compressed_sparse_row_graph<Directed, VertexProperty, EdgeProperty,
                                       GraphProperty, Vertex, EdgeIndex>;

} // namespace pcolorizer