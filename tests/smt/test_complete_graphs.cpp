#include "testing/generators.h"
#include <gtest/gtest.h>

#include "smt.h"

TEST(SMTCompleteGraph, K3) {
  static constexpr std::size_t V = 3;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraph, K4) {
  static constexpr std::size_t V = 4;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraph, K6) {
  static constexpr std::size_t V = 6;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraph, K8) {
  static constexpr std::size_t V = 8;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraph, K16) {
  static constexpr std::size_t V = 16;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraph, Large) {
  static constexpr std::size_t V = 256;
  auto graph = generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto [chi, _] = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}