#include "testing/generators.h"
#include <gtest/gtest.h>

#include "smt.h"

TEST(SMTCycleGraph, C3) {
  static constexpr std::size_t V = 3;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 3);
}

TEST(SMTCycleGraph, C4) {
  static constexpr std::size_t V = 4;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C6) {
  static constexpr std::size_t V = 6;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C8) {
  static constexpr std::size_t V = 8;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C9) {
  static constexpr std::size_t V = 9;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 3);
}

TEST(SMTCycleGraph, C10) {
  static constexpr std::size_t V = 10;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C12) {
  static constexpr std::size_t V = 12;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C14) {
  static constexpr std::size_t V = 14;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, C16) {
  static constexpr std::size_t V = 16;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}

TEST(SMTCycleGraph, Large) {
  static constexpr std::size_t V = 256;
  auto graph = generators::make_cycle_graph(V);

  pcolorizer::SmtColorizer colorizer;
  auto chi = colorizer.min_colors(graph);
  EXPECT_EQ(chi, 2);
}