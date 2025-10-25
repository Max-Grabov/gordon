#pragma once

#include <cstdint>
#include <string>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

namespace Gordon
{
namespace ImageUtils
{
namespace mp = ::mediapipe;
namespace tasks = mp::tasks;
namespace vision = mp::tasks::vision;

static const int HAND_CONN[][2] = {
  {0,1},{1,2},{2,3},{3,4},
  {0,5},{5,6},{6,7},{7,8},        
  {5,9},{9,10},{10,11},{11,12},   
  {9,13},{13,14},{14,15},{15,16}, 
  {13,17},{17,18},{18,19},{19,20},
  {0,17} 
};

static const int HAND_CONN_N = sizeof(HAND_CONN)/sizeof(HAND_CONN[0]);

static const int POSE_CONN[][2] = {
  {11,12},{11,13},{13,15},{12,14},{14,16},
  {15,17},{15,19},{15,21},{16,18},{16,20},{16,22},
  {11,23},{12,24},{23,24},
  {23,25},{25,27},{27,29},{29,31},
  {24,26},{26,28},{28,30},{30,32},
  {0,11},{0,12}
};

static const int POSE_CONN_N = sizeof(POSE_CONN)/sizeof(POSE_CONN[0]);

mp::Image makeSolidRGBImage(const int &width, const int &height, const uint8_t &r, const uint8_t &g, const uint8_t &b);

bool loadImageAsMP(const std::string& file_path, mp::Image* output, cv::Mat* background_output);

inline int clampi(const int &value, const int &low, const int high); 

cv::Mat generatePoseImage(const std::string &image_path, const std::string &hand_task, const std::string &pose_task);
}
}
