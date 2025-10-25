#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace mp = mediapipe;
namespace tasks = mp::tasks;
namespace vision = mp::tasks::vision;

static mp::Image MakeSolidRGBImage(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
  mp::ImageFrame frame(mp::ImageFormat::SRGB, w, h, mp::ImageFrame::kDefaultAlignmentBoundary);
  uint8_t* data = frame.MutablePixelData();
  const int stride = frame.WidthStep();
  for (int y = 0; y < h; ++y) {
    uint8_t* row = data + y * stride;
    for (int x = 0; x < w; ++x) {
      row[3*x+0] = r; row[3*x+1] = g; row[3*x+2] = b;
    }
  }
  return mp::Image(std::make_shared<mp::ImageFrame>(std::move(frame)));
}

static bool LoadImageAsMP(const std::string& path, mp::Image* out, cv::Mat* bgr_out) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) return false;
  cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  mp::ImageFrame frame(mp::ImageFormat::SRGB, rgb.cols, rgb.rows, mp::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat dst = mediapipe::formats::MatView(&frame);
  rgb.copyTo(dst);
  *out = mp::Image(std::make_shared<mp::ImageFrame>(std::move(frame)));
  *bgr_out = bgr.clone();                 
  return true;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static const int HAND_CONN[][2] = {
  {0,1},{1,2},{2,3},{3,4},        // thumb
  {0,5},{5,6},{6,7},{7,8},        // index
  {5,9},{9,10},{10,11},{11,12},   // middle
  {9,13},{13,14},{14,15},{15,16}, // ring
  {13,17},{17,18},{18,19},{19,20},// pinky
  {0,17}                           // palm base
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

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << " smoke <image_path> <hand.task> <pose.task>\n";
    return 1;
  }

  std::string image_path, hand_task, pose_task;
  if (argc == 4) { image_path = argv[1]; hand_task = argv[2]; pose_task = argv[3]; }
  else { hand_task = argv[1]; pose_task = argv[2]; }

  mp::Image image;
  cv::Mat canvas_bgr;
  if (!image_path.empty()) {
    if (!LoadImageAsMP(image_path, &image, &canvas_bgr)) {
      std::cerr << "failed  " << image_path << "\n"; return 1;
    }
  } else {
    image = MakeSolidRGBImage(512, 512, 128, 128, 128);
    canvas_bgr = cv::Mat(image.height(), image.width(), CV_8UC3, cv::Scalar(128,128,128));
  }
  const int W = image.width(), H = image.height();
  std::cout << "image size: " << W << "x" << H << "\n";

  auto hand_opts = std::make_unique<vision::hand_landmarker::HandLandmarkerOptions>();
  hand_opts->base_options.model_asset_path = hand_task;
  hand_opts->num_hands = 2;
  auto hand_or = vision::hand_landmarker::HandLandmarker::Create(std::move(hand_opts));
  if (!hand_or.ok()) { std::cerr << "hand create: " << hand_or.status() << "\n"; return 1; }
  auto hand_res = hand_or.value()->Detect(image);
  if (!hand_res.ok()) { std::cerr << "hand detect: " << hand_res.status() << "\n"; return 1; }

  auto pose_opts = std::make_unique<vision::pose_landmarker::PoseLandmarkerOptions>();
  pose_opts->base_options.model_asset_path = pose_task;
  pose_opts->num_poses = 1;
  auto pose_or = vision::pose_landmarker::PoseLandmarker::Create(std::move(pose_opts));
  if (!pose_or.ok()) { std::cerr << "pose create: " << pose_or.status() << "\n"; return 1; }
  auto pose_res = pose_or.value()->Detect(image);
  if (!pose_res.ok()) { std::cerr << "pose detect: " << pose_res.status() << "\n"; return 1; }

  for (size_t i = 0; i < hand_res->hand_landmarks.size(); ++i) {
    const auto& nlm = hand_res->hand_landmarks[i].landmarks;
    std::cout << "hand[" << i << "] landmarks: " << nlm.size() << "\n";
    for (size_t j = 0; j < nlm.size(); ++j) {
      std::cout << "  (" << j << "): x=" << nlm[j].x << " y=" << nlm[j].y << " z=" << nlm[j].z << "\n";
    }
  }

  for (size_t i = 0; i < pose_res->pose_landmarks.size(); ++i) {
    const auto& nlm = pose_res->pose_landmarks[i].landmarks;
    std::cout << "pose[" << i << "] landmarks: " << nlm.size() << "\n";
    for (size_t j = 0; j < nlm.size(); ++j) {
      std::cout << "  (" << j << "): x=" << nlm[j].x << " y=" << nlm[j].y << " z=" << nlm[j].z << "\n";
    }
  }

  auto draw_circle = [&](float nx, float ny, int r=3) {
    int x = clampi((int)std::round(nx * (W - 1)), 0, W - 1);
    int y = clampi((int)std::round(ny * (H - 1)), 0, H - 1);
    cv::circle(canvas_bgr, {x, y}, r, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
  };
  auto draw_line = [&](float nx1, float ny1, float nx2, float ny2, int t=2) {
    int x1 = clampi((int)std::round(nx1 * (W - 1)), 0, W - 1);
    int y1 = clampi((int)std::round(ny1 * (H - 1)), 0, H - 1);
    int x2 = clampi((int)std::round(nx2 * (W - 1)), 0, W - 1);
    int y2 = clampi((int)std::round(ny2 * (H - 1)), 0, H - 1);
    cv::line(canvas_bgr, {x1, y1}, {x2, y2}, cv::Scalar(0, 200, 255), t, cv::LINE_AA);
  };

  for (size_t i = 0; i < hand_res->hand_landmarks.size(); ++i) {
    const auto& nlm = hand_res->hand_landmarks[i].landmarks;
    for (const auto& p : nlm) draw_circle(p.x, p.y, 3);
    if (nlm.size() >= 21) {
      for (int k = 0; k < HAND_CONN_N; ++k) {
        const auto& a = nlm[HAND_CONN[k][0]];
        const auto& b = nlm[HAND_CONN[k][1]];
        draw_line(a.x, a.y, b.x, b.y, 2);
      }
    }
  }

  for (size_t i = 0; i < pose_res->pose_landmarks.size(); ++i) {
    const auto& nlm = pose_res->pose_landmarks[i].landmarks;
    for (const auto& p : nlm) draw_circle(p.x, p.y, 2);
    if (nlm.size() >= 33) {
      for (int k = 0; k < POSE_CONN_N; ++k) {
        const auto& a = nlm[POSE_CONN[k][0]];
        const auto& b = nlm[POSE_CONN[k][1]];
        draw_line(a.x, a.y, b.x, b.y, 2);
      }
    }
  }

  std::string out_path = image_path.empty() ? "test/out.png" : (image_path + ".overlay.png");
  if (!cv::imwrite(out_path, canvas_bgr)) {
    std::cerr << "failed to write " << out_path << "\n";
  } else {
    std::cout << "\nSaved overlay: " << out_path << "\n";
  }

  return 0;
}
