#pragma once

#include <boost/graph/adjacency_list.hpp>

namespace pcolorizer {

template <typename OutEdgeListS = boost::vecS,
          typename VertexListS = boost::vecS,
          typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property,
          typename GraphProperty = boost::no_property,
          typename EdgeListS = boost::listS>
using UndirectedGraph =
    boost::adjacency_list<OutEdgeListS, VertexListS, boost::undirectedS,
                          VertexProperty, EdgeProperty, GraphProperty,
                          EdgeListS>;

} // namespace pcolorizer