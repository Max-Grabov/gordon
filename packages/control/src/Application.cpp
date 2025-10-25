#include "Application.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include "Mirroring.hpp"
#include "Recording.hpp"

namespace Gordon
{
Application::Application() : running_(false)
{
	// Load all of the maps whatever we need later
	std::cout << "sup\n";
	control_modes_.emplace("Recording", std::make_unique<Recording>());
	control_modes_.emplace("Mirroring", std::make_unique<Mirroring>());
}

Application::~Application()
{
	std::cout << "destroyed\n";
}

void handleEvents()
{
	// TODO GET THE STRING FROM SOMEWHERE!
	
	std::string event{"Mirror"};
	if(event == "Idle")
	{
		//DO IDLE
	}
	else if(event == "Mirror")
	{
		camera_event_queue_->push("Mirror");
	}
	else if(event == "Record")
	{
		camera_event_queue_->push("Record");
	}
	else if(event == "Exit")
	{
		camera_event_queue_->push("Exit");
		running = false_;
	}
}

void Application::run()
{

	// Launch Thread for Camera using jthread
	
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

	control_modes_["Recording"]->run();
	control_modes_["Mirroring"]->run();
}
}

