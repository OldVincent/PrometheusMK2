#include "FrameCountService.hpp"

#include <iostream>

namespace RoboPioneers::Sparrow
{
	/// 构造函数
	FrameCountService::FrameCountService() : PeriodBeginTime(std::chrono::steady_clock::now())
	{}

	/// 更新事件
	void FrameCountService::OnUpdate(Frame &frame)
	{
		auto current_time = std::chrono::steady_clock::now();

		++CurrentCount;

		if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - PeriodBeginTime) >
			CurrentPeriodSpan)
		{
			// 计算帧速率
			FrameRateSource = 1000 / CurrentPeriodSpan.count() * CurrentCount;
			// 记录总帧数
			LastPeriodFrameCountSource = CurrentCount;

			// 重置统计帧数
			CurrentCount = 0;

			// 更新时间周期
			CurrentPeriodSpan = PeriodSpan;
			// 重设周期起始时间
			PeriodBeginTime = current_time;

			// 若开启了自动输出，则自动打印帧速率
			if (AutoPrint)
			{
				std::cout << "FPS: " << FrameRate << std::endl;
			}
		}
	}
}