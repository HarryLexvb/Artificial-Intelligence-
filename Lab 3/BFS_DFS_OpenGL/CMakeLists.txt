cmake_minimum_required(VERSION 3.24)
project(BFS_DFS_OpenGL)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#link sfml libraries
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
#link glfw libraries
find_package(glfw3 3.3 REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})
include_directories(${GLFW_INCLUDE_DIR})

add_executable(BFS_DFS_OpenGL main.cpp ${SOURCES})

target_link_libraries(BFS_DFS_OpenGL ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
target_link_libraries(BFS_DFS_OpenGL sfml-graphics sfml-audio)
target_link_libraries(BFS_DFS_OpenGL glfw ${GLFW_LIBRARIES})
