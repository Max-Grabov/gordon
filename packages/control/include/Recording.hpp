#pragma once

#include "Mode.hpp"
#include "ThreadSafeQueue.hpp"
#include <memory>

namespace Gordon
{
class Recording : public Mode
{
public:
	Recording();
	~Recording();
	Recording &operator=(const Recording &other) = delete;
	Recording &operator=(const Recording &&other) = delete;
	Recording(const Recording &&other) = delete;
	Recording(const Recording &other) = delete;

	void run() override;
private:
	std::shared_ptr<ThreadSafeQueue<std::shared_ptr<std::string>>> camera_event_queue_;
	bool running_{false};
};
}
