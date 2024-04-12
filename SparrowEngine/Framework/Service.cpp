#include "Service.hpp"

namespace RoboPioneers::Sparrow
{
	/// 更新方法
	void Service::Update(Frame &frame)
	{
		if (Enable)
		{
			OnUpdate(frame);
		}
	}
}