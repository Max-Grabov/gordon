cmake_minimum_required(VERSION 3.20)

project(gordon LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- Paths ----------
set(MP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe")

# ---------- Source globs (non-recursive) ----------
file(GLOB FRAMEWORK_FORMATS_SRC_FILES       CONFIGURE_DEPENDS "${MP_ROOT}/framework/formats/*.cc")
file(GLOB FRAMEWORK_PORT_SRC_FILES          CONFIGURE_DEPENDS "${MP_ROOT}/framework/port/*.cc")
file(GLOB TASKS_CC_CORE_SRC_FILES           CONFIGURE_DEPENDS "${MP_ROOT}/tasks/cc/core/*.cc")
file(GLOB TASKS_CC_VISION_HAND_LM_SRC_FILES CONFIGURE_DEPENDS "${MP_ROOT}/tasks/cc/vision/hand_landmarker/*.cc")
file(GLOB TASKS_CC_VISION_POSE_LM_SRC_FILES CONFIGURE_DEPENDS "${MP_ROOT}/tasks/cc/vision/pose_landmarker/*.cc")
file(GLOB GPU_SRC_FILES                     CONFIGURE_DEPENDS "${MP_ROOT}/gpu/*.cc")

# ---------- Dependencies ----------
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(Protobuf REQUIRED)  # protoc + runtime
# Protobuf imported target name normalization
if (TARGET protobuf::libprotobuf)
  set(PROTOBUF_LIB protobuf::libprotobuf)
elseif (TARGET protobuf::protobuf)
  set(PROTOBUF_LIB protobuf::protobuf)
else()
  message(FATAL_ERROR "Found Protobuf, but no imported runtime target. Install libprotobuf-dev.")
endif()

# Eigen (header-only)
find_package(Eigen3 REQUIRED NO_MODULE)
# If your distro lacks Eigen3::Eigen target, fallback:
# find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
# if (EIGEN3_INCLUDE_DIR) list(APPEND EXTRA_PUBLIC_INCLUDES "${EIGEN3_INCLUDE_DIR}") endif()

# Abseil (vendored)
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/abseil-cpp")

# ---------- TensorFlow Lite (C++) ----------
include(FetchContent)
# Configure these if desired:
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "Enable XNNPACK for TFLite" FORCE)
set(TFLITE_TAG v2.16.1 CACHE STRING "TensorFlow tag for TFLite CMake build")

FetchContent_Declare(
  tensorflow_src
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        ${TFLITE_TAG}
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tensorflow_src)
if (NOT tensorflow_src_POPULATED)
  FetchContent_Populate(tensorflow_src)
endif()

# Build only TFLite via its CMakeLists
# It creates target: tensorflow-lite
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK ${TFLITE_ENABLE_XNNPACK} CACHE BOOL "" FORCE)
add_subdirectory("${tensorflow_src_SOURCE_DIR}/tensorflow/lite"
                 "${tensorflow_src_BINARY_DIR}/tflite-build")

# ---------- Robust protoc detection ----------
set(PROTOC "")
if (TARGET Protobuf::protoc)
  set(PROTOC $<TARGET_FILE:Protobuf::protoc>)
elseif (TARGET protobuf::protoc)
  set(PROTOC $<TARGET_FILE:protobuf::protoc>)
elseif (DEFINED Protobuf_PROTOC_EXECUTABLE AND Protobuf_PROTOC_EXECUTABLE)
  set(PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
else()
  find_program(PROTOC NAMES protoc)
  if(NOT PROTOC)
    message(FATAL_ERROR "Could not find protoc. Install protobuf-compiler.")
  endif()
endif()

# ---------- Protobuf generation: ONLY exact directories (no recursion) ----------
set(PROTO_DIRS
  "${MP_ROOT}/framework/formats"
  "${MP_ROOT}/framework/port"
  "${MP_ROOT}/tasks/cc/core"
  "${MP_ROOT}/tasks/cc/vision/hand_landmarker"
  "${MP_ROOT}/tasks/cc/vision/pose_landmarker"
  "${MP_ROOT}/gpu"
)

# Collect *.proto (NON-recursive) and de-dup
set(MP_PROTOS_UNIQ "")
foreach(PDIR IN LISTS PROTO_DIRS)
  if (EXISTS "${PDIR}")
    file(GLOB FOUND_PROTO_IN_DIR
         LIST_DIRECTORIES FALSE
         CONFIGURE_DEPENDS
         "${PDIR}/*.proto")
    foreach(p IN LISTS FOUND_PROTO_IN_DIR)
      get_filename_component(rp "${p}" REALPATH)
      list(FIND MP_PROTOS_UNIQ "${rp}" _idx)
      if (_idx EQUAL -1)
        list(APPEND MP_PROTOS_UNIQ "${rp}")
      endif()
    endforeach()
  endif()
endforeach()

# protoc -I search list: ONLY your exact dirs + the repo root that contains "mediapipe/"
set(PROTOC_IMPORTS "")
list(APPEND PROTOC_IMPORTS -I "${CMAKE_CURRENT_SOURCE_DIR}/lib")
foreach(IDIR IN LISTS PROTO_DIRS)
  if (EXISTS "${IDIR}")
    list(APPEND PROTOC_IMPORTS -I "${IDIR}")
  endif()
endforeach()

# Output dir for generated code; mirror layout relative to MP_ROOT
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(GEN_SRCS "")
set(GEN_HDRS "")

foreach(proto ${MP_PROTOS_UNIQ})
  get_filename_component(PDIR  "${proto}" DIRECTORY)
  get_filename_component(PNAME "${proto}" NAME_WE)
  file(RELATIVE_PATH RELDIR "${MP_ROOT}" "${PDIR}")
  set(OUT_DIR "${GEN_DIR}/${RELDIR}")

  set(OUT_CC "${OUT_DIR}/${PNAME}.pb.cc")
  set(OUT_H  "${OUT_DIR}/${PNAME}.pb.h")

  add_custom_command(
    OUTPUT "${OUT_CC}" "${OUT_H}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
    COMMAND "${PROTOC}"
            ${PROTOC_IMPORTS}
            --cpp_out "${GEN_DIR}"
            "${proto}"
    DEPENDS "${proto}"
    COMMENT "Generating ${RELDIR}/${PNAME}.pb.*"
    VERBATIM
  )

  list(APPEND GEN_SRCS "${OUT_CC}")
  list(APPEND GEN_HDRS "${OUT_H}")
endforeach()

# ---------- Library ----------
add_library(gordon STATIC
  ${FRAMEWORK_FORMATS_SRC_FILES}
  ${FRAMEWORK_PORT_SRC_FILES}
  ${TASKS_CC_CORE_SRC_FILES}
  ${TASKS_CC_VISION_HAND_LM_SRC_FILES}
  ${TASKS_CC_VISION_POSE_LM_SRC_FILES}
  ${GPU_SRC_FILES}
  ${GEN_SRCS}  # generated .pb.cc
  ${GEN_HDRS}  # for IDE visibility
)

target_include_directories(gordon PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/lib"  # so #include "mediapipe/..." works
  "${MP_ROOT}"
  "${GEN_DIR}"                       # generated headers
  ${OpenCV_INCLUDE_DIRS}
  # ${EIGEN3_INCLUDE_DIR}            # only if you used the fallback find_path
)

target_link_libraries(gordon
  PRIVATE
    ${PROTOBUF_LIB}
    glog::glog
    absl::log
    Eigen3::Eigen
    tensorflow-lite         # from the TFLite subdir build
    ${OpenCV_LIBS}
    dl pthread m            # typical TFLite/system deps
)

# ---------- Executable ----------
add_executable(Gordon "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(Gordon PRIVATE gordon)
