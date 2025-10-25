#pragma once

#include "ImageUtils.hpp"

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
//	std::unordered_map<std::string, Modes> control_modes;
	bool running_{false};
};
} 
