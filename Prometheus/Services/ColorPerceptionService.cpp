#include "ColorPerceptionService.hpp"

#include "../Modules/CUDAUtility.hpp"
#include "../Modules/ImageDebugUtility.hpp"

namespace RoboPioneers::Prometheus
{
	//==============================
	// 处理过程
	//==============================

	/// 筛选红色
	cv::cuda::GpuMat ColorPerceptionService::FilterRedArea(const cv::cuda::GpuMat& hsv_picture) const
	{
		std::vector<cv::cuda::GpuMat> channels;
		cv::cuda::split(hsv_picture, channels);

		// 第一区段色调蒙版
		cv::cuda::GpuMat hue1;
		cv::cuda::threshold(channels[0], hue1, Settings.RedThresholds.Hue1UpperBound, 255,
					  cv::THRESH_BINARY_INV);

		// 第二区段色调蒙版
		cv::cuda::GpuMat hue2;
		cv::cuda::threshold(channels[0], hue2, Settings.RedThresholds.Hue2LowerBound, 255,
					  cv::THRESH_BINARY);

		// 饱和度蒙版
		cv::cuda::GpuMat saturation;
		cv::cuda::threshold(channels[1], saturation, Settings.RedThresholds.SaturationLowerBound, 255,
					  cv::THRESH_BINARY);

		// 值蒙版
		cv::cuda::GpuMat value;
		cv::cuda::threshold(channels[2], value, Settings.RedThresholds.ValueLowerBound, 255,
					  cv::THRESH_BINARY);

		// 制作蒙版
		cv::cuda::GpuMat mask;
		cv::cuda::bitwise_or(hue1, hue2, mask, saturation);
		cv::cuda::bitwise_and(mask, value, mask);

		return mask;
	}

	/// 筛选蓝色区域
	cv::cuda::GpuMat ColorPerceptionService::FilterBlueArea(const cv::cuda::GpuMat& hsv_picture) const
	{
		std::vector<cv::cuda::GpuMat> channels;
		cv::cuda::split(hsv_picture, channels);

		// 色调蒙版
		cv::cuda::GpuMat hue;
		cv::cuda::threshold(channels[0], hue, Settings.BlueThresholds.HueLowerBound,
					  255, cv::THRESH_BINARY);
		cv::cuda::threshold(hue, hue, Settings.BlueThresholds.HueUpperBound,
					  255, cv::THRESH_BINARY_INV);

		// 饱和度蒙版
		cv::cuda::GpuMat saturation;
		cv::cuda::threshold(channels[1], saturation, Settings.BlueThresholds.SaturationLowerBound,
					  255, cv::THRESH_BINARY);

		// 亮度蒙版
		cv::cuda::GpuMat value;
		cv::cuda::threshold(channels[2], value, Settings.BlueThresholds.ValueLowerBound,
					  255, cv::THRESH_BINARY);

		cv::cuda::GpuMat mask;
		cv::cuda::bitwise_and(hue, value, mask, saturation);

		return mask;
	}

	/// 更新方法
	void ColorPerceptionService::OnUpdate(Sparrow::Frame &frame)
	{
		if (Settings.TargetColor == TargetColorEnum::Red)
		{
			/// 筛选红色区域
			Output.MaskPicture = FilterRedArea(frame.GpuPicture);
		}
		else if (Settings.TargetColor == TargetColorEnum::Blue)
		{
			/// 筛选蓝色区域
			Output.MaskPicture = FilterBlueArea(frame.GpuPicture);
		}

		/// 应用开运算过滤器
		Properties.CloseFilter->apply(Output.MaskPicture, Output.MaskPicture);

		/// 等待设备运算结束
		Modules::CUDAUtility::SynchronizeDevice();

		#ifdef DEBUG
		Modules::ImageDebugUtility::ShowGPUPicture("Color Perception", Output.MaskPicture);
		#endif
	}
}
