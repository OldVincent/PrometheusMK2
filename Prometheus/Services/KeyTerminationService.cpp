#include "KeyTerminationService.hpp"

namespace RoboPioneers::Prometheus
{
	/// 更新方法
	void KeyTerminationService::OnUpdate(Sparrow::Frame &frame)
	{
		if (cv::waitKey(Settings.WaitMilliseconds) == Settings.CloseKeyASCII)
		{
			if (Input.TargetLifeFlag)
			{
				Input.TargetLifeFlag->store(false);
			}
		}
	}
}
