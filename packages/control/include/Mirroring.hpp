#pragma once

#include "Mode.hpp"

namespace Gordon
{
class Mirroring : public Mode
{
public:
	Mirroring();
	~Mirroring();
	Mirroring &operator=(const Mirroring &other) = delete;
	Mirroring &operator=(const Mirroring &&other) = delete;
	Mirroring(const Mirroring &&other) = delete;
	Mirroring(const Mirroring &other) = delete;

	void run() override;
private:
	std::shared_ptr<ThreadSafeQueue<std::shared_ptr<std::string>>> camera_event_queue_;
	bool running_{false};
};
}
