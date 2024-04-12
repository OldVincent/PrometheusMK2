#pragma once

#include <chrono>
#include <unordered_map>
#include <opencv4/opencv2/opencv.hpp>

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief 帧信息类
	 * @author Vincent
	 * @details
	 *  ~ 该类用于传递帧信息。
	 */
	class Frame
	{
	private:
		/// 当前帧时间
		std::chrono::time_point<std::chrono::steady_clock> CurrentTimeSource;
		/// 帧时间间隔
		std::chrono::milliseconds DeltaTimeSource;

	public:
		//==============================
		// 构造函数与析构函数部分
		//==============================

		/// 构造函数
		Frame();

		//==============================
		// 帧信息部分
		//==============================

		/// 当前帧时间，即当前帧被创建的时间，获取自steady_clock
		const decltype(CurrentTimeSource)& CurrentTime {CurrentTimeSource};
		/// 帧间隔时间，即两帧之间间隔的时间，单位为微秒
		const decltype(DeltaTimeSource)& DeltaTime {DeltaTimeSource};

		/// 存储在显存中的图像，HSV格式
		cv::cuda::GpuMat GpuPicture;

		/// 坐标偏移量
		cv::Point2i PointOffset;

		#ifdef DEBUG
		/// 原始图像
		cv::Mat OriginalPicture;
		cv::Mat CutPicture;
		#endif

		//==============================
		// 控制方法部分
		//==============================

		/**
		 * @brief 重设帧信息
		 * @param picture 内存图片的右值引用
		 * @param gpu_picture 显存图片的右值引用
		 * @details
		 *  ~ 将重设图片、帧创建时间，并计算帧间隔时间。
		 */
		void Reset(cv::Mat&& picture, const cv::cuda::GpuMat& gpu_picture);
	};
}
