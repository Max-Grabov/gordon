#include "Application.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace Gordon
{
Application::Application() : running_(false)
{
	// Load all of the maps whatever we need later
	std::cout << "sup\n";
}

Application::~Application()
{
	std::cout << "destroyed\n";
}

void Application::run()
{
	// Test for making the image to get bazel to build
	std::cout << "Generating le image\n";
	std::string input{"sample_images/gettyimages-1685801220-612x612.jpg"};
	std::string hand_task{"lib/mediapipe/test/models/hand_landmarker.task"};
	std::string pose_task{"lib/mediapipe/test/models/pose_landmarker_full.task"};

	const auto output = ImageUtils::generatePoseImage(input, hand_task, pose_task);

	std::string output_path = input + ".epic_output.png";

	if (!cv::imwrite(output_path, output)) {
    std::cerr << "failed to write " << output_path << "\n";
  } else {
    std::cout << "Saved overlay: " << output_path << "\n";
  }
}
}
