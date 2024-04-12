#include "PictureCuttingService.hpp"

#include <utility>
#include "../Modules/CUDAUtility.hpp"
#include "../Modules/ImageDebugUtility.hpp"

namespace RoboPioneers::Prometheus
{
	/// 更新方法
	void PictureCuttingService::OnUpdate(Sparrow::Frame &frame)
	{
		if (*Input.NeedToCut && !Settings.ForceNotCut)
		{
			frame.GpuPicture = CutPictureByInterestedRegion(&frame.GpuPicture,
			                             *Input.CuttingArea);
			frame.PointOffset = cv::Point{(Input.CuttingArea->x, Input.CuttingArea->y)};
		}
		else
		{
			frame.PointOffset = cv::Point{0,0};
		}

		#ifdef DEBUG
		frame.GpuPicture.download(frame.CutPicture);
		Modules::CUDAUtility::SynchronizeDevice();
		cv::imshow("Cutting Picture", frame.CutPicture);
		#endif
	}

	/// 使用兴趣区方式裁剪图像
	cv::cuda::GpuMat PictureCuttingService::CutPictureByInterestedRegion(cv::cuda::GpuMat const *picture, cv::Rect area)
	{
		cv::cuda::GpuMat target(area.height, area.width, picture->type());
		(*picture)(std::move(area)).copyTo(target);
		return target;
	}
}
