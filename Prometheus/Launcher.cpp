#include "Controller.hpp"

#include <SparrowEngine/SparrowEngine.hpp>

int main(int arguments_count, char** arguments)
{
	using namespace RoboPioneers::Prometheus;
	using namespace RoboPioneers::Sparrow;

	Controller controller;

	Engine->Launch(arguments_count, arguments, &controller);
}
