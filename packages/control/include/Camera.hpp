#pragma once

#include "ThreadSafeQueue.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace Gordon
{
class Camera
{
public:
	namespace cvVidCap = cv::VideoCapture;

	Camera(cvVidCap &camera_capture, const std::string &source_file);

	~Camera();

	Camera(const Camera &other) = delete;
	
	Camera(const Camera &&other) = delete;

	Camera &operator=(const Camera &other) = delete;

	Camera &operator=(const Camera &&other) = delete;

	void run();

private:
	bool openCameraStream();

	void openWindow();

	void handleEvents();

	std::weak_ptr<ThreadSafeQueue<std::shared_ptr<std::string>>> event_queue_;
	cvVidCap camera_capture_;
	std::string source_file_;
	bool valid_{false};
};
}
