cmake_minimum_required(VERSION 3.10)
project(TP_contours LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(edge_eval
    src/main.cpp
    src/detectors.cpp
    src/scoring.cpp
    src/analysis.cpp)
target_include_directories(edge_eval PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(edge_eval PRIVATE ${OpenCV_LIBS})

# Project folder, so data/ and results/ are found whatever the launch directory
target_compile_definitions(edge_eval PRIVATE "PROJECT_DIR=\"${CMAKE_SOURCE_DIR}\"")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edge_eval PRIVATE -Wall -Wextra)
endif()
