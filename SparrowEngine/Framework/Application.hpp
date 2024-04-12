#pragma once

#include <CameraDriver/CameraDriver.hpp>
#include <SerialPortDriver/SerialPortDriver.hpp>

#include <string>
#include <vector>
#include <atomic>
#include <list>
#include <memory>

#include "../Drivers/HSVDualMatAcquisitor.hpp"

#include "Frame.hpp"

#include "../Services/FrameTimeControlService.hpp"
#include "../Services/FrameCountService.hpp"

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief 应用接口
	 * @author Vincent
	 * @details
	 *  ~ 该类定义了基本的程序生命周期控制方法。
	 */
	class Application
	{
		/// 声明运行时类为友元类，由运行时控制应用的基本生命周期
		friend class Runtime;

	private:
		//==============================
		// 设备设定信息
		//==============================

		/// 相机对象在相机列表中的索引
		unsigned int CameraIndex;
		/// 串口文件名称
		std::string SerialPortName;

		//==============================
		// 生命周期相关对象
		//==============================

		/// 当前帧对象
		Frame CurrentFrame;

	protected:
		//==============================
		// 设备控制对象
		//==============================

		/// 内置设备
		struct {
			/// 相机对象
			Modules::CameraDriver::CameraDevice Camera;
			/// 相机图像采集器
			HSVDualMatAcquisitor Acquisitor {&Camera};
			/// 串口对象
			Modules::SerialPortDriver::SerialPort Port;
		}InnerDevices;

		//==============================
		// 生命周期控制对象
		//==============================

		/// 内置设定
		struct
		{
			/**
		 	 * @brief 生命旗标
		 	 * @details
		 	 *  ~ 被运行时启动后，该旗标的值被设置为true。
		 	 *  ~ 将该旗标的值设置为false，将结束自身的生命周期，从而使得程序进入退出流程。
		 	 */
			std::atomic_bool LifeFlag {false};
			/// 串口功能开关，便于测试
			bool EnableSerialPort {true};
			/// 调试功能开关
			bool EnableDebug {false};
		}InnerSettings;

		//==============================
		// 内置服务对象
		//==============================

		/// 内置服务
		struct {
			/// 帧时间控制单元
			FrameTimeControlService FrameTimeController;
			/// 帧速率计算单元
			FrameCountService FrameRateCounter;
		}InnerServices;

	public:
		//==============================
		// 构造函数与析构函数
		//==============================

		/**
		 * @brief 构造函数
		 * @param camera_index 相机设备索引
		 * @param serial_port_file 串口设备文件
		 */
		Application(unsigned int camera_index, std::string serial_port_file);

	private:
		//==============================
		// 供引擎调用的私有生命阶段方法
		// 这些方法是供引擎调用以进行生命周期的。
		//==============================

		/**
		 * @brief 安装设备方法
		 * @details
		 *  ~ 该方法将准备各个设备对象直至所有设备对象都开始正常工作。
		 */
		void Install();

		/**
		 * @brief 卸载设备方法
		 * @details
		 *  ~ 该方法将停止正在工作的设备，并进行一定的清理工作。
		 */
		void Uninstall();

		/**
		 * @brief 更新方法
		 * @details
		 *  ~ 将从摄像头中读取图像，调用用户的更新方法，随后将结果送入串口。
		 */
		void Update();

		//==============================
		// 供应用接口调用的事件
		//==============================

		/**
		 * @brief 安装内置服务事件
		 * @details
		 *  ~ 该事件内将安装内置服务。
		 */
		void OnInstallInnerServices();

		/**
		 * @brief 用户服务更新前事件
		 * @param frame 帧信息对象
		 * @details
		 *  ~ 该事件将在用户的服务更新前被调用。
		 */
		void OnBeforeUserServices(const Frame& current_frame);

		/**
		 * @brief 用户服务更新后时间
		 * @param frame 帧信息对象
		 * @details
		 *  ~ 该事件将在用户的服务更新后被调用。
		 */
		void OnAfterUserServices(const Frame& current_frame);

		/**
		 * @brief 更新后事件
		 * @param frame 帧信息对象
		 * @details
		 *  ~ 该事件将在用户的服务更新后调用。
		 */
		void OnAfterUpdate(Frame &current_frame);

		/**
		 * @brief 卸载内置服务事件
		 * @details
		 *  ~ 该事件内将卸载内置服务。
		 */
		void OnUninstallInnerServices();

	protected:
		//==============================
		// 供用户派生的生命阶段方法
		// 这些方法是供用户派生以实现额外操作的。
		//==============================

		/**
		 * @brief 安装设备事件
		 * @details
		 *  ~ 该方法会在安装设备阶段被调用。
		 *  ~ 可以用来安装额外的设备。
		 */
		virtual void OnInstallDevices() {};

		/**
		 * @brief 配置设备事件
		 * @details
		 *  ~ 该方法会在设备均被正确安装后调用。
		 *  ~ 可以在该方法中设置摄像头和串口的参数。
		 *  ~ 该方法被调用后，启动这些设备的方法将被引擎调用。
		 */
		virtual void OnConfigureDevices() {};

		/**
		 * @brief 安装服务事件
		 * @return 将要在更新时被更新的服务
		 * @details
		 *  ~ 该方法将在各个设备正常运转的时候被调用。
		 *  ~ 该事件成功触发并成功结束后，将进入应用的主循环。
		 */
		virtual void OnInstallServices() {};

		/**
		 * @brief 卸载服务事件
		 * @details
		 *  ~ 该事件将在应用的主循环结束的时候被调用。
		 *  ~ 该事件结束后，将进入应用的卸载设备阶段。
		 */
		virtual void OnUninstallServices() {};

		/**
		 * @brief 卸载设备事件
		 * @details
		 *  ~ 该方法会在卸载设备阶段被调用。
		 *  ~ 可以用来卸载和清理额外的设备。
		 */
		virtual void OnUninstallDevices() {};

		/**
		 * @brief 更新时间
		 * @param frame 帧信息对象
		 * @return 要输入到串口的字节向量
		 */
		virtual std::vector<unsigned char> OnUpdate(Frame &frame) = 0;
	};
}
