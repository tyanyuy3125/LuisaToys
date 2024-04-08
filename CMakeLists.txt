cmake_minimum_required(VERSION 3.5.0)
project(lct VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_executable(lct main.cpp)

add_library(luisa-render-include INTERFACE)
target_include_directories(luisa-render-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(luisa-render-include INTERFACE c_std_11 cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(luisa-render-include INTERFACE
            /wd4068 # Maxwell loves custom pragmas
            /wd5105 # winbase.h: macro expansion producing 'defined' has undefined behavior
            /Zc:preprocessor)
endif ()

set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(LUISA_COMPUTE_ENABLE_UNITY_BUILD OFF CACHE BOOL "" FORCE)
add_subdirectory(compute)
target_include_directories(luisa-render-include INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/compute/include/luisa)

target_link_libraries(lct PUBLIC luisa::compute luisa-render-include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
