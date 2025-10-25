#pragma once

namespace Gordon
{

class Mode
{
public:
	virtual void run() = 0;
	virtual ~Mode() = default;
};
}
