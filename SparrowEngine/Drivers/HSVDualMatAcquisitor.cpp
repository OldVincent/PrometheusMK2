#include "HSVDualMatAcquisitor.hpp"

#include <DxImageProc.h>
#include <opencv4/opencv2/cudaimgproc.hpp>

extern void CUDADeviceSynchronize();

namespace RoboPioneers::Sparrow
{

	void HSVDualMatAcquisitor::ReceivePictureIncomeEvent(
			Modules::CameraDriver::Acquisitors::AbstractAcquisitor::RawPicture data)
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
		          RAW2RGB_NEIGHBOUR, BAYERBG, false);

		std::unique_lock lock(PictureMutex);

		GpuPicture.upload(picture);
		Picture = picture;
		cv::cuda::cvtColor(GpuPicture, GpuPicture, cv::COLOR_BGR2HSV);

		CUDADeviceSynchronize();
	}
}