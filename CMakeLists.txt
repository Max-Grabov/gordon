cmake_minimum_version(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

project(gordon)

file(SRC_FILES CONFIGURE_DEPENDS
  ${SRC_DIR}/Application.cpp
  ${SRC_DIR}/Mirroring.cpp
  ${SRC_DIR}/Recording.cpp
  ${SRC_DIR}/ImageUtils.cpp
  ${SRC_DIR}/Camera.cpp
)

add_library(gordon STATIC ${SRC_FILES})

target_include_directories(gordon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(Gordon ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(Gordon gordon)
