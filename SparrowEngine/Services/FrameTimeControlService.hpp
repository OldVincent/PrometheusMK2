#pragma once

#include "../Framework/Service.hpp"

#include <chrono>

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief 帧时间控制服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务将在当前帧时间小于目标帧时间时进行等待。
	 */
	class FrameTimeControlService : public Service
	{
	public:
		/**
		 * @brief 目标帧时间
		 * @details
		 *  ~ 为0表示不限制。
		 *  ~ 将根据该帧率计算帧时间，若上一帧时间比目标帧时间短，则将等待缺少的时间。
		 */
		std::chrono::milliseconds TargetFrameTime {0};

	protected:
		/// 更新事件
		void OnUpdate(Frame &frame) override;
	};
}
