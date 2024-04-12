#pragma once

namespace RoboPioneers::Modules
{
	/**
	 * @brief CUDA实用工具模块
	 * @author Vincent
	 * @details
	 *  ~ 该模块提供了对CUDA简单常用的控制方法。
	 */
	class CUDAUtility
	{
	public:
		/// 同步设备
		static void SynchronizeDevice();
	};
}