#pragma once

#include <boost/concept_check.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>

namespace pcolorizer {

namespace concepts {

template <typename G>
class VertexColorableGraph : boost::concepts::VertexListGraph<G>,
                             boost::concepts::EdgeListGraph<G>,
                             boost::concepts::IncidenceGraph<G> {
public:
  typedef typename boost::graph_traits<G>::directed_category directed_category;

  BOOST_CONCEPT_USAGE(VertexColorableGraph) {
    BOOST_CONCEPT_ASSERT(
        (boost::Convertible<directed_category, boost::undirected_tag>));
  }
};

} // namespace concepts

} // namespace pcolorizer