#include "Recording.hpp"

#include <iostream>

namespace Gordon
{
Recording::Recording() : running_(true) {}

Recording::~Recording() { std::cout << "Destroying Recording\n"; }
void Recording::run()
{
	std::cout << "Running recording\n";
}
}
