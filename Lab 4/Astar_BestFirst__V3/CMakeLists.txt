cmake_minimum_required(VERSION 3.23)
project(Astar_BestFirst__V3)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#link sfml libraries
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})

add_executable(Astar_BestFirst__V3 main.cpp)


target_link_libraries(Astar_BestFirst__V3 ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(Astar_BestFirst__V3 sfml-graphics)
