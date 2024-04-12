#include "MatAcquisitor.hpp"

#include <DxImageProc.h>
#include <stdexcept>
#include <thread>

namespace RoboPioneers::Modules::CameraDriver::Acquisitors
{
	/// 接受采集到图片的事件
	void MatAcquisitor::ReceivePictureIncomeEvent(AbstractAcquisitor::RawPicture data)
	{
		if (!IsDeviceWorking.load())
		{
			IsDeviceWorking = true;
		}

		// 更新最新图片状态
		if (!IsPictureLatest.load())
		{
			IsPictureLatest = true;
		}

		auto picture = cv::Mat(cv::Size(data.Width, data.Height), CV_8UC3);
		DxRaw8toRGB24(const_cast<void *>(data.Data), picture.data,
		              static_cast<VxUint32>(data.Width), static_cast<VxUint32>(data.Height),
		              RAW2RGB_NEIGHBOUR, BAYERBG, false);

		std::unique_lock lock(PictureMutex);

		Picture = std::move(picture);
	}

	/// 接受到设备离线事件
	void MatAcquisitor::ReceiveDeviceOfflineEvent()
	{
		IsDeviceWorking = false;
	}

	/// 获取新采集的图片
	cv::Mat MatAcquisitor::GetPicture(bool wait_for_latest)
	{
		if (!IsStarted())
		{
			throw std::logic_error("MatAcquisitor::GetPicture Collection Has Not Been Started.");
		}

		if (!IsWorking())
		{
			throw std::runtime_error("MatAcquisitor::GetPicture Device Is Offline.");
		}

		// 如果要求等待，则会不断地核验图片是否为最新
		while (wait_for_latest && !IsPictureLatest.load())
		{
			// 等待时交出当前的时间片，处理器会将分配给该线程资源暂时交给别的县城
			std::this_thread::yield();
		}

		std::shared_lock lock(PictureMutex);
		// 若图像为空，说明这是第一次获取图像，需要等待第一张图像到达
		while (Picture.empty())
		{
			lock.unlock();
			std::this_thread::yield();
			lock.lock();
		}
		IsPictureLatest = false;
		return Picture;
	}
}