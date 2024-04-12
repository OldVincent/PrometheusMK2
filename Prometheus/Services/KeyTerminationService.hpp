#pragma once

#include <SparrowEngine/SparrowEngine.hpp>

#include <atomic>

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 按键终止服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于检测用户按键，允许用户在按下指定按键后结束程序。
	 */
	class KeyTerminationService : public Sparrow::Service
	{
	public:
		// 输入
		struct {
			/**
			 * @brief 目标生命旗标
			 * @details 在指定的检测周期内检测到关闭按键按下后，将关闭该生命旗标
			 */
			std::atomic<bool> *TargetLifeFlag;
		}Input;

		// 设定
		struct {
			/// 关闭按键的ASCII
			char CloseKeyASCII {27};
			/// 按键等待时间，单位为微秒，设置为0时将无限等待
			int WaitMilliseconds {1};
		}Settings;

	protected:
		/// 更新方法
		void OnUpdate(Sparrow::Frame &frame) override;
	};
}
