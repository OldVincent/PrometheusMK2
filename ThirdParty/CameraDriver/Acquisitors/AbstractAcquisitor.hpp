#pragma once

#include "../CameraDevice.hpp"

#include <atomic>

namespace RoboPioneers::Modules::CameraDriver::Acquisitors
{
	/**
	 * @brief 抽象采集器
	 * @author Vincent
	 * @details
	 *  ~ 该采集器接受相机的回调，但却不进行处理。
	 *  ~ 用户可以派生该采集器以实现自定义的处理逻辑。
	 */
	class AbstractAcquisitor
	{
	public:
		/**
		 * @brief 构造函数
		 * @param device 相机设备对象指针
		 */
		explicit AbstractAcquisitor(CameraDevice* device);

		/// 析构函数，若设备未关闭，则将自动关闭设备
		virtual ~AbstractAcquisitor();

		/// 采集器是否已经开始采集
		std::atomic_bool IsCollectorStarted {false};

	private:
		/// 设备指针
		CameraDevice* Device;

	protected:
		/// 设备离线事件句柄，用于在停止采集时注销该事件
		void* OfflineEventHandle {nullptr};

		/// 设备是否正在工作，即采集图片
		std::atomic_bool IsDeviceWorking {false};
	public:
		/**
		 * @brief 获取设备对象指针
		 * @return 相机设备对象指针
		 */
		[[nodiscard]] CameraDevice* GetDevice() const noexcept
		{
			return Device;
		}

		//==============================
		// 基本控制方法
		//==============================

		/**
		 * @brief 开始采集
		 * @throw std::runtime_error 当设备指针或句柄为空或控制命令执行失败
		 * @pre CameraDevice已经被打开
		 */
		virtual void Start();

		/**
		 * @brief 停止采集
		 * @throw std::runtime_error 当设备指针或句柄为空或控制命令执行失败
		 * @pre Start()方法已经被调用
		 */
		virtual void Stop();

		//==============================
		// 基本查询方法
		//==============================

		/**
		 * @brief 查询采集器是否已经开始采集
		 * @return 若已经被要求开始采集，则返回true；若Start()方法未被调用或Stop()方法已被调用，返回false
		 */
		[[nodiscard]] bool IsStarted() const noexcept
		{
			return IsCollectorStarted;
		}

		/**
		 * @brief 查询采集器是否正在工作
		 * @return 若已经开始采集且状态正常，则返回true；若未开始采集或采集过程中设备离线，返回false。
		 */
		[[nodiscard]] bool IsWorking() const noexcept
		{
			return IsDeviceWorking;
		}

		//==============================
		// 事件处理方法
		//==============================

		/**
		 * @brief 原始图片数据
		 * @details
		 *  ~ 该结构体用于存储相机采集的原始数据的内存指针及相关信息。
		 *  ~ 对于大恒水星系列相机而言，图像格式默认为BayerRG8。
		 */
		class RawPicture
		{
		public:
			/**
			 * @brief 构造函数
			 * @param data 图片数据指针
			 * @param width 图片的宽度，即横向像素点个数
			 * @param height 图片的高度，即纵向像素点个数
			 */
			RawPicture(void* data, int width ,int height) :
				Data(data), Width(width), Height(height)
			{}

			/// 数据指针，格式为BayerRG
			void* Data;
			/// 图像宽度
			int Width;
			/// 图像高度
			int Height;
		};

		/**
		 * @brief 接收到图片事件
		 * @param data 原始图片数据
		 * @details
		 *  ~ 当采集到图片并处理完毕时，该方法将被采集线程调用。
		 */
		virtual void ReceivePictureIncomeEvent(RawPicture data) = 0;

		/**
		 * @brief 接受到设备离线事件
		 * @details
		 *  ~ 当设备离线时，该方法将被采集线程调用。
		 */
		virtual void ReceiveDeviceOfflineEvent() = 0;
	};
}