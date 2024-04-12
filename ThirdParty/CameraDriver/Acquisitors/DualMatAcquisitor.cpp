#include "DualMatAcquisitor.hpp"

#ifndef NO_CUDA

#include <DxImageProc.h>
#include <thread>

extern void CUDADeviceSynchronize();

namespace RoboPioneers::Modules::CameraDriver::Acquisitors
{
	void DualMatAcquisitor::ReceivePictureIncomeEvent(AbstractAcquisitor::RawPicture data)
	{
		// 更新设备工作状态
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
		              RAW2RGB_ADAPTIVE, BAYERBG, false);

		std::unique_lock lock(PictureMutex);

		GpuPicture.upload(picture);
		Picture = picture;

		CUDADeviceSynchronize();
	}

	/// 获取图片
	std::tuple<cv::Mat, cv::cuda::GpuMat> DualMatAcquisitor::GetDualPicture(bool wait_for_latest)
	{
		if (!IsStarted())
		{
			throw std::logic_error("DualMatAcquisitor::GetPicture Collection Has Not Been Started.");
		}

		if (!IsWorking())
		{
			throw std::runtime_error("DualMatAcquisitor::GetPicture Device Is Offline.");
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
		return {Picture, GpuPicture};
	}


}

#endif
