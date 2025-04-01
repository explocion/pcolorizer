#include "testing/dimacs.h"
#include <gtest/gtest.h>

#include "smt.h"

#include <fstream>
#include <iostream>

TEST(SMTDimacsGraph, DSJC250_1) {
  std::ifstream dimacs("instances/DSJC250.1.col");
  auto graph = dimacs::read_dimacs_coloring(dimacs).value();

  pcolorizer::SmtColorizer colorizer;
  auto [chi, timeout] = colorizer.min_colors(graph);
  std::cout << "DSJC250.1: " << chi;
  if (timeout)
    std::cout << " (timeout)";
  std::cout << std::endl;
}

TEST(SMTDimacsGraph, DSJC1000_9) {
  std::ifstream dimacs("instances/DSJC1000.9.col");
  auto graph = dimacs::read_dimacs_coloring(dimacs).value();

  pcolorizer::SmtColorizer colorizer;
  auto [chi, timeout] = colorizer.min_colors(graph);
  std::cout << "DSJC250.1: " << chi;
  if (timeout)
    std::cout << " (timeout)";
  std::cout << std::endl;
}
