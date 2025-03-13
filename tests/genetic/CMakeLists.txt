add_executable(test_genetic test_complete_graph.cu)
target_link_libraries(test_genetic PRIVATE GTest::gtest_main)
set_target_properties(test_genetic PROPERTIES CUDA_ARCHITECTURES native)

add_test(NAME test_genetic COMMAND test_genetic)
