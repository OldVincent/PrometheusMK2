#include "Frame.hpp"

#include <utility>

namespace RoboPioneers::Sparrow
{
	/// 构造函数
	Frame::Frame() : CurrentTimeSource(std::chrono::steady_clock::now()), DeltaTimeSource()
	{}

	/// 重设图片
	void Frame::Reset(cv::Mat&& picture, const cv::cuda::GpuMat& gpu_picture)
	{
		GpuPicture = gpu_picture;

		#ifdef DEBUG
		OriginalPicture = picture;
		#endif

		auto last_frame_time = CurrentTimeSource;
		CurrentTimeSource = std::chrono::steady_clock::now();
		DeltaTimeSource = std::chrono::duration_cast<std::chrono::milliseconds>(CurrentTimeSource - last_frame_time);
	}
}