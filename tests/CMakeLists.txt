find_package(GTest REQUIRED)

include_directories(include)

enable_testing()
add_subdirectory(smt)
add_subdirectory(genetic)
add_subdirectory(meta)
