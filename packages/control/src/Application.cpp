#include "Application.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include "Mirroring.hpp"
#include "Recording.hpp"

namespace Gordon
{
Application::Application() : camera_(cvVidCap()), camera_event_queue_(std::make_shared<EventQueue>()), state_("Idle"), running_(false)
{
	// Load all of the maps whatever we need later
	std::cout << "Constructed\n";

	control_modes_.emplace("Record", std::make_unique<Recording>());
	control_modes_.emplace("Mirror", std::make_unique<Mirroring>());
}

Application::~Application()
{
	std::cout << "Destroyed\n";
}

void handleEvents(const std::string &event)
{
	// TODO GET THE STRING FROM SOMEWHERE NOT PARAMETER!
	
	if(!events_.contains(event))
	{
		state_ = "Exit";
	}
	else
	{		
		state_ = event;
	}
}

void processState()
{
	camera_event_queue->push(event);

	// Before started next application loop, store a conditional variable
	// Camera will send a signal when it is ready, for maximum safety - Max
	if(state_ == "Exit")
	{
		running_ = false;
		return;
	}
	else if(state_ == "Idle")
	{
		continue;
	}
	
	// We are guarranteed to have a valid state at this point, so we can just go into the map
	control_modes_[state_]->run();
}
void Application::run()
{	
	// Launch Thread for Camera using jthread TODO
	std::string event{"Mirror"};
	while(1)
	{	
		// FOR NOW TAKE IN A STRING BUT WE SHOULD GET THE STRING FROM VOICE PYTHON TBA
		handleEvents(event);
		processState();
	}
}
}

