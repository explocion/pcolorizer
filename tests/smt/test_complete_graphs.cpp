#include "testing/generators.h"
#include <gtest/gtest.h>

#include "smt.h"

TEST(SMTCompleteGraphs, K3) {
  static constexpr std::size_t V = 3;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K4) {
  static constexpr std::size_t V = 4;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K6) {
  static constexpr std::size_t V = 6;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K8) {
  static constexpr std::size_t V = 8;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K10) {
  static constexpr std::size_t V = 10;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K12) {
  static constexpr std::size_t V = 12;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K14) {
  static constexpr std::size_t V = 14;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}

TEST(SMTCompleteGraphs, K16) {
  static constexpr std::size_t V = 16;
  auto graph = pcolorizer::generators::make_complete_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, V);
}