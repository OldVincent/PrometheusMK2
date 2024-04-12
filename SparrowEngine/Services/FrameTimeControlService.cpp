#include "FrameTimeControlService.hpp"

#include <thread>

namespace RoboPioneers::Sparrow
{
	/// 更新事件
	void FrameTimeControlService::OnUpdate(Frame &frame)
	{
		// 等待满足帧时间
		std::this_thread::sleep_for(std::chrono::steady_clock::now() - frame.CurrentTime);
	}
}