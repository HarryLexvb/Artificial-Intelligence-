cmake_minimum_required(VERSION 3.25)
project(TSP_GeneticAlgorithm)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#link sfml libraries
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
#link QT
find_package(Qt5 COMPONENTS Charts REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})

add_executable(TSP_GeneticAlgorithm main.cpp header.h)

target_link_libraries(TSP_GeneticAlgorithm ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(TSP_GeneticAlgorithm sfml-graphics)
target_link_libraries(TSP_GeneticAlgorithm Qt5::Charts)
