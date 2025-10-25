#include "Mirroring.hpp"

#include <iostream>

namespace Gordon
{
Mirroring::Mirroring() : running_(true) {}

Mirroring::~Mirroring() { std::cout << "Destroying Mirroring\n"; }
void Mirroring::run()
{
	std::cout << "Running mirroring\n";
}
}
