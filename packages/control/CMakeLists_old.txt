cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gordon)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB FRAMEWORK_FORMATS_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/framework/formats/*.cc")
file(GLOB FRAMEWORK_PORT_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/framework/port/*.cc")
file(GLOB TASKS_CC_CORE_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/core/*.cc")
file(GLOB TASKS_CC_VISION_HAND_LANDMARKER_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/vision/hand_landmarker/*.cc")
file(GLOB TASKS_CC_VISION_POSE_LANDMARKER_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/vision/pose_landmarker/*.cc")
file(GLOB GPU_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/vision/pose_landmarker/*.cc")

find_package(OpenCV REQUIRED)

find_package(glog REQUIRED)

# TODO WHEN WE MOVE TO THE NANO, MAKE SURE ABSL INSTALLS TO THE SAME PLACE, OR ADD IT HERE - MAX
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lib/abseil-cpp")

add_library(gordon STATIC ${FRAMEWORK_FORMATS_SRC_FILES}
													${FRAMEWORK_PORT_SRC_FILES}
													${TASKS_CC_CORE_SRC_FILES}
													${TASKS_CC_VISION_HAND_LANDMARKER_SRC_FILES}
													${TASKS_CC_VISION_POSE_LANDMARKER_SRC_FILES}
													${GPU_SRC_FILES}
)

target_include_directories(gordon PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/lib
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/framework/formats
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/framework/port
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/core
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/vision/hand_landmarker
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/tasks/cc/vision/pose_landmarker
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/gpu
	${CMAKE_CURRENT_SOURCE_DIR}/lib/mediapipe/gpu
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(gordon ${OpenCV_LIBS}
														 absl::log
														 glog::glog
)

add_executable(Gordon ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(Gordon gordon)


