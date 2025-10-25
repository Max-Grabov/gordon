#include "ImageUtils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace Gordon
{
namespace ImageUtils
{
mp::Image makeSolidRGBImage(const int &width, const int &height, const uint8_t &r, const uint8_t &g, const uint8_t &b)
{
	mp::ImageFrame frame(mp::ImageFormat::SRGB, width, height, mp::ImageFrame::kDefaultAlignmentBoundary);
  uint8_t* data{frame.MutablePixelData()};
  const int stride{frame.WidthStep()};
  for (int y = 0; y < height; ++y) {
    uint8_t* row = data + y * stride;
    for (int x = 0; x < width; ++x) {
      row[3*x+0] = r; 
			row[3*x+1] = g; 
			row[3*x+2] = b;
    }
  }
  return mp::Image(std::make_shared<mp::ImageFrame>(std::move(frame)));
}

bool loadImageAsMP(const std::string& file_path, mp::Image* output, cv::Mat* background_output)
{
	cv::Mat background{cv::imread(file_path, cv::IMREAD_COLOR)};
  if (background.empty())
	{
		return false;
	}

  cv::Mat rgb; 
	cv::cvtColor(background, rgb, cv::COLOR_BGR2RGB);

  mp::ImageFrame frame{mp::ImageFormat::SRGB, rgb.cols, rgb.rows, mp::ImageFrame::kDefaultAlignmentBoundary};
  cv::Mat destination{mediapipe::formats::MatView(&frame)};
  rgb.copyTo(destination);

  *output = mp::Image(std::make_shared<mp::ImageFrame>(std::move(frame)));
  *background_output = background.clone();

  return true;
}

inline int clampi(const int &value, const int &low, const int high)
{
	return value < low ? low : (value > high ? high : value);
}

cv::Mat generatePoseImage(const std::string &image_path, const std::string &hand_task, const std::string &pose_task)
{	
	mp::Image image;
	cv::Mat canvas_bgr;
	if (!image_path.empty()) {
		if (!loadImageAsMP(image_path, &image, &canvas_bgr)) {
			std::cerr << "failed  " << image_path << "\n"; return cv::Mat();
		}
	} else {
		image = makeSolidRGBImage(512, 512, 128, 128, 128);
		canvas_bgr = cv::Mat(image.height(), image.width(), CV_8UC3, cv::Scalar(128,128,128));
	}
	const int W = image.width(), H = image.height();
	std::cout << "image size: " << W << "x" << H << "\n";

	auto hand_opts = std::make_unique<vision::hand_landmarker::HandLandmarkerOptions>();
	hand_opts->base_options.model_asset_path = hand_task;
	hand_opts->num_hands = 2;
	auto hand_or = vision::hand_landmarker::HandLandmarker::Create(std::move(hand_opts));
	if (!hand_or.ok()) { std::cerr << "hand create: " << hand_or.status() << "\n"; return cv::Mat(); }
	auto hand_res = hand_or.value()->Detect(image);
	if (!hand_res.ok()) { std::cerr << "hand detect: " << hand_res.status() << "\n"; return cv::Mat(); }

	auto pose_opts = std::make_unique<vision::pose_landmarker::PoseLandmarkerOptions>();
	pose_opts->base_options.model_asset_path = pose_task;
	pose_opts->num_poses = 1;
	auto pose_or = vision::pose_landmarker::PoseLandmarker::Create(std::move(pose_opts));
	if (!pose_or.ok()) { std::cerr << "pose create: " << pose_or.status() << "\n"; return cv::Mat(); }
	auto pose_res = pose_or.value()->Detect(image);
	if (!pose_res.ok()) { std::cerr << "pose detect: " << pose_res.status() << "\n"; return cv::Mat(); }

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

	return canvas_bgr;
}
}
}
