#pragma once

#include "Mode.hpp"

namespace Gordon
{
class Recording : Mode
{
public:
	Recording();
	~Recording();
	Recording &operator=(const Recording &other) = delete;
	Recording &operator=(const Recording &&other) = delete;
	Recording(const Recording &&other) = delete;
	Recording(const Recording &other) = delete;

	void run();
private:
	bool running_{false};
};
}
