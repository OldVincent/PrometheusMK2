#include "CameraDevice.hpp"

#include <GxIAPI.h>
#include <DxImageProc.h>
#include <stdexcept>

namespace RoboPioneers::Modules::CameraDriver
{
	class GalaxySDKAutoManager
	{
	public:
		/// 构造函数，将初始化GalaxySDK库
		GalaxySDKAutoManager() noexcept
		{
			GXInitLib();
		}
	};

	/// 构造函数
	CameraDevice::CameraDevice() : DeviceHandle(nullptr)
	{
		[[maybe_unused]] static GalaxySDKAutoManager GalaxySDKAutoManagerInstance;
	}

	/// 打开相机
	void CameraDevice::Open(unsigned int index)
	{
		std::unique_lock lock(DeviceHandleMutex);

		// 操作执行结果
		GX_STATUS operation_result;

		// 获取设备列表
		uint32_t device_count = 0;
		operation_result = GXUpdateDeviceList(&device_count, 500);
		if (operation_result != GX_STATUS_LIST::GX_STATUS_SUCCESS)
		{
			throw std::runtime_error("CameraDevice::Open Failed to Query Device List.");
		}
		if (device_count <= 0)
		{
			throw std::runtime_error("CameraDevice::Open No Camera Detected.");
		}
		if (device_count <= index)
		{
			throw std::runtime_error("CameraDevice::Open Invalid Device Index.");
		}

		// 开启指定设备，注意，GalaxySDK中开启设备的该方法中设备索引是从1开始编号的
		operation_result = GXOpenDeviceByIndex(index + 1, &DeviceHandle);
		if (operation_result != GX_STATUS_LIST::GX_STATUS_SUCCESS)
		{
			throw std::runtime_error("CameraDevice::Open Failed to Open Device.");
		}
	}

	/// 关闭相机
	void CameraDevice::Close()
	{
		std::unique_lock lock(DeviceHandleMutex);

		if (DeviceHandle)
		{
			GXCloseDevice(DeviceHandle);
			DeviceHandle = nullptr;
		}
	}

	/// 设置曝光时间
	bool CameraDevice::SetExposureTime(double value)
	{
		if (DeviceHandle &&
			GXSetFloat(DeviceHandle, GX_FLOAT_EXPOSURE_TIME, value) == GX_STATUS_LIST::GX_STATUS_SUCCESS)
		{
			return true;
		}
		return false;
	}

	/// 设置增益
	bool CameraDevice::SetGain(double value)
	{
		if (DeviceHandle &&
		    GXSetFloat(DeviceHandle, GX_FLOAT_GAIN, value) == GX_STATUS_LIST::GX_STATUS_SUCCESS)
		{
			return true;
		}
		return false;
	}

	/// 设置白平衡的值
	bool CameraDevice::SetWhiteBalance(CameraDevice::WhiteBalanceChannel channel, double value)
	{
		if (DeviceHandle)
		{
			GX_STATUS operation_result;

			switch (channel)
			{
				case WhiteBalanceChannel::Red:
					operation_result = GXSetEnum(DeviceHandle, GX_ENUM_BALANCE_RATIO_SELECTOR,
								  GX_BALANCE_RATIO_SELECTOR_RED);
					break;
				case WhiteBalanceChannel::Green:
					operation_result = GXSetEnum(DeviceHandle, GX_ENUM_BALANCE_RATIO_SELECTOR,
								  GX_BALANCE_RATIO_SELECTOR_GREEN);
					break;
				case WhiteBalanceChannel::Blue:
					operation_result = GXSetEnum(DeviceHandle, GX_ENUM_BALANCE_RATIO_SELECTOR,
								  GX_BALANCE_RATIO_SELECTOR_BLUE);
					break;
			}
			if (operation_result != GX_STATUS_LIST::GX_STATUS_SUCCESS)
			{
				throw std::runtime_error("CameraDevice::SetWhiteBalance Failed to Switch White Balance Channel.");
			}

			operation_result = GXSetFloat(DeviceHandle, GX_FLOAT_BALANCE_RATIO, value);
			if (operation_result != GX_STATUS_LIST::GX_STATUS_SUCCESS)
			{
				throw std::runtime_error("CameraDevice::SetWhiteBalance Failed to Set White Balance Ratio.");
			}

			return true;
		}

		return false;
	}
}

