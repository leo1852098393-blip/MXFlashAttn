cmake_minimum_required(VERSION 3.20)
project(mxflashattn_native LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

add_library(_C MODULE attention_forward.cpp paged_kv.cpp workspace.cpp)
target_compile_features(_C PRIVATE cxx_std_17)
target_include_directories(_C PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} Python::Module)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
set_target_properties(_C PROPERTIES PREFIX "" OUTPUT_NAME "_C" SUFFIX ".${Python_EXTENSION_SUFFIX}")
