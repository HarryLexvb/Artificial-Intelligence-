cmake_minimum_required(VERSION 3.23)
project(Astar_BestFisrt)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#link sfml libraries
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})

add_executable(Astar_BestFisrt main.cpp ${SOURCES} Graph.cpp Graph.h Grid.cpp Grid.h Algorithm.h Location.h AlgorithmFactory.h AStar.cpp AStar.h BestFS.cpp BestFS.h Game.cpp Game.h)

target_link_libraries(Astar_BestFisrt ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(Astar_BestFisrt sfml-graphics)

