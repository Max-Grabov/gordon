#include "Camera.hpp"

namespace Gordon
{
Camera::Camera(cvVidCap &camera_capture, const std::string &source_file) : camera_capture_(std::move(camera_capture)), source_file_(std::move(source_file)), valid_(true)
{
	if(!openCameraStream());
	{
		valid_ = false;
		return;
	}

	openWindow();
}

void run()
{
	if(!valid)
	{
		return;
	}

	cv::Mat frame;

	while(1)
	{
		if (!camera_capture_.read(frame) || frame.empty()) {
      std::cerr << "End of stream or failed read\n";
      break;
    }

		handleEvents();

		// TODO POLL EVENTS VIA MY BEAUTIFUL THREAD SAFE QUEUE
		// MAKE EVENT CLASS?
		cv::imshow("stream", frame);
	}
}

bool handleEvents()
{
	if(event_queue_->empty())
	{
		return false;
	}

	const auto event = event_queue_->wait_and_pop();
	if(!event)
	{
		return true;
	}

}

bool Camera::openCameraStream()
{
	if(!valid_)
	{
		return false
	}

	char *end{'\0'};
	long id{std::strtol(source_file.c_str(), &end, 10)};

	if(!end)
	{
		return camera_capture_.open(static_cast<int>(id), cv::CAP_ANY);
	}

	return camera_capture_.open(source_file, cv::CAP_ANY);
}

void Camera::openWindow()
{
  camera_capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  camera_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  camera_capture_.set(cv::CAP_PROP_FPS, 30);

  cv::namedWindow("Camera_Stream", cv::WINDOW_AUTOSIZE);
}

Camera::~Camera()
{
	camera_capture_.release();
	cv::destroyAllWindows();
}
}
