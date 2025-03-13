add_executable(test_meta test_complete_graph.cu)
target_link_libraries(test_meta PRIVATE GTest::gtest_main)
set_target_properties(test_meta PROPERTIES CUDA_ARCHITECTURES native)

add_test(NAME test_meta COMMAND test_meta)
