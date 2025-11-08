#pragma once

#include "ImageUtils.hpp"
#include "ThreadSafeQueue.hpp"
#include "Mode.hpp"
#include "Camera.hpp"
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace Gordon
{
class Application
{
public:
	Application();

	Application(const Application &other) = delete;

	Application &operator=(const Application &other) = delete;

	Application(const Application &&other) = delete;

	Application &operator=(const Application &&other) = delete;

	~Application();

	void run();
private:
	void handleEvents();

	void processState();

	Camera camera_;
	std::unordered_set<std::string> events_{"Idle", "Exit", "Mirror", "Running"};
	std::shared_ptr<ThreadSafeQueue<std::shared_ptr<std::string>>> camera_event_queue_;
  std::unordered_map<std::string, std::unique_ptr<Mode>> control_modes_;
	std::string state_;
	bool running_{false};
};
} 
