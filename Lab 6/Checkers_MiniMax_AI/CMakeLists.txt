cmake_minimum_required(VERSION 3.23)
project(Checkers_MiniMax_AI)

set(CMAKE_CXX_STANDARD 23)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#link sfml libraries
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})

add_executable(Checkers_MiniMax_AI main.cpp board.cpp board.h game.cpp game.h pawn.cpp pawn.h)

target_link_libraries(Checkers_MiniMax_AI ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(Checkers_MiniMax_AI sfml-graphics)
