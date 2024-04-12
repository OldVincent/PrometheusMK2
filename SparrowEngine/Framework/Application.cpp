#include "Application.hpp"

#include <thread>
#include <iostream>
#include <utility>

namespace RoboPioneers::Sparrow
{
	//==============================
	// 构造函数与析构函数
	//==============================

	/// 构造函数
	Application::Application(unsigned int camera_index, std::string serial_port_file):
			CameraIndex(camera_index), SerialPortName(std::move(serial_port_file))
	{}

	//==============================
	// 生命阶段方法
	//==============================

	/// 安装设备方法
	void Application::Install()
	{
		if (InnerSettings.EnableSerialPort)
		{
			/*
		     * 打开串口
		     * 串口打开失败一般是串口文件名写错了，直接让程序结束就可以了。
		     */
			InnerDevices.Port.Open(SerialPortName);
		}

		/*
		 * 打开相机
		 * 经常性地会出现相机暂时离线的情况，比如开机时相机线未正确链接等。
		 * 若此处直接让程序结束，则需要重启程序或计算机，较为麻烦，故此处等待相机上限。
		 */
		bool camera_ready = false;
		while(!camera_ready)
		{
			try
			{
				InnerDevices.Camera.Open(CameraIndex);
				camera_ready = true;
			}catch(std::runtime_error& error)
			{
				std::cerr << "Failed to Open Camera: " << error.what() << std::endl;
				std::cerr << "Will Retry in 30 seconds." << std::endl;

				std::this_thread::sleep_for(std::chrono::seconds(30));
			}
		}

		// 调用用户的安装设备方法
		OnInstallDevices();

		// 调用用户的配置设备方法
		OnConfigureDevices();

		// 启动采集器，开始采集图像
		InnerDevices.Acquisitor.Start();

		// 安装内置服务
		OnInstallInnerServices();

		// 调用用户的安装服务方法
		OnInstallServices();
	}

	/// 卸载设备方法
	void Application::Uninstall()
	{
		// 调用用户的卸载服务方法
		OnUninstallServices();

		// 卸载内置服务
		OnUninstallInnerServices();

		// 停止采集器
		InnerDevices.Acquisitor.Stop();

		// 调用用户的卸载设备方法
		OnUninstallDevices();

		// 关闭相机
		InnerDevices.Camera.Close();

		if (InnerSettings.EnableSerialPort)
		{
			// 关闭串口
			InnerDevices.Port.Close();
		}
	}

	/// 更新方法
	void Application::Update()
	{
		// 获取图像
		auto [picture, gpu_picture] = InnerDevices.Acquisitor.GetDualPicture(true);

		// 重设帧
		CurrentFrame.Reset(std::move(picture), gpu_picture);

		// 触发用户服务更新前事件
		OnBeforeUserServices(CurrentFrame);

		// 获取数据
		auto data = OnUpdate(CurrentFrame);

		// 触发用户服务更新后事件
		OnAfterUserServices(CurrentFrame);

		if (InnerSettings.EnableSerialPort)
		{
			// 写入
			InnerDevices.Port.Write(data);
		}

		// 触发更新后事件
		OnAfterUpdate(CurrentFrame);
	}

	//==============================
	// 内置服务适配部分
	//==============================

	/// 安装内置服务事件
	void Application::OnInstallInnerServices()
	{
	}

	/// 卸载内置服务事件
	void Application::OnUninstallInnerServices()
	{

	}

	/// 用户服务更新前事件
	void Application::OnBeforeUserServices(const Frame &current_frame)
	{

	}

	/// 用户服务更新后事件
	void Application::OnAfterUserServices(const Frame &current_frame)
	{

	}

	/// 更新后事件
	void Application::OnAfterUpdate(Frame &current_frame)
	{
		// 更新帧时间控制器
		InnerServices.FrameTimeController.Update(current_frame);
		// 更新帧速率计算器
		InnerServices.FrameRateCounter.Update(current_frame);
	}
}