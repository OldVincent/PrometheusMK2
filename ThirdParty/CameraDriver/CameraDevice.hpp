#pragma once

#include <atomic>
#include <shared_mutex>

namespace RoboPioneers::Modules::CameraDriver
{
	/**
	 * @brief 相机设备类
	 * @author Vincent
	 * @details
	 *  ~ 该类提供对相机参数的基本设置和基本控制。
	 *  ~ 该类采取装饰器模式，仅预留采集接口，不提供具体的采集结果处理操作。
	 *  ~ 具体的采集结果处理操作为操作器的工作。
	 */
	class CameraDevice
	{
	protected:
		/// 设备句柄互斥量
		mutable std::shared_mutex DeviceHandleMutex;
		/// 设备句柄
		void* DeviceHandle;

	public:

		//==============================
		// 构造与析构函数部分
		//==============================

		/// 构造函数
		CameraDevice();
		/// 析构函数
		virtual ~CameraDevice() = default;

		//==============================
		// 相机基本控制部分
		//==============================

		/**
		 * @brief 打开相机
		 * @param index 相机在列表中的索引，从0开始
		 */
		virtual void Open(unsigned int index);

		/**
		 * @brief 关闭相机
		 */
		virtual void Close();

		/**
		 * @brief 获取设备句柄
		 * @return 用于与GalaxySDK通信的设备句柄
		 */
		[[nodiscard]] void* GetDeviceHandle() const noexcept
		{
			std::shared_lock lock(DeviceHandleMutex);
			return DeviceHandle;
		}

		/**
		 * @brief 判断是否是否已经开启
		 * @return
		 */
		[[nodiscard]] bool IsOpened() const noexcept
		{
			return DeviceHandle != nullptr;
		}

		//==============================
		// 相机参数设置部分
		//==============================

		/**
		 * @brief 设置曝光时间
		 * @param value 曝光时间，单位为微秒(us)
		 * @return 是否操作成功，操作成功则返回true，失败则返回false
		 */
		bool SetExposureTime(double value);

		/**
		 * @brief 设置增益
		 * @param value 增益，单位为db
		 * @return 是否操作成功，操作成功则返回true，失败则返回false
		 */
		bool SetGain(double value);

		/**
		 * @brief 白平衡通道
		 */
		enum class WhiteBalanceChannel
		{
			/// 红色通道
			Red,
			/// 绿色通道
			Green,
			/// 蓝色通道
			Blue
		};

		/**
		 * @brief 设置白平衡值
		 * @param channel 白平衡通道
		 * @param value 指定通道上的白平衡的值
		 * @return 是否操作成功，操作成功则返回true，失败则返回false
		 */
		bool SetWhiteBalance(WhiteBalanceChannel channel, double value);
	};
}
