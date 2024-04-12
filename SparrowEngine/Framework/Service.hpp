#pragma once

#include "Frame.hpp"

#include <atomic>

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief 服务接口
	 * @author Vincent
	 * @details
	 *  ~ 服务是用于实现某一特定目的的逻辑聚合体。
	 *  ~ 服务的设定由全局变量进行，输入输出使用指针的方式进行绑定。
	 *  ~ 服务通过被每帧调用的Update方法执行功能。
	 */
	class Service
	{
	public:
		/**
		 * @brief 是否可用，即功能开关
		 * @details
		 *  ~ 将其设置为false，则再调用Update方法时，其不会触发用户的OnUpdate事件。
		 */
		std::atomic_bool Enable {true};

		/**
		 * @brief 更新方法
		 * @param frame 帧信息
		 * @details
		 *  ~ 一般情况下该方法会被应用每帧调用。
		 */
		void Update(Frame &frame);

	protected:
		/**
		 * @brief 更新事件
		 * @param frame 帧信息
		 * @details
		 *  ~ 当服务的Enable为true的时候该方法会被每帧调用。
		 */
		virtual void OnUpdate(Frame &frame) = 0;
	};
}