#pragma once

#ifndef NO_CUDA

#include "MatAcquisitor.hpp"

namespace RoboPioneers::Modules::CameraDriver::Acquisitors
{
	/**
	 * @brief cv::GpuMat格式采集器
	 * @author Vincent
	 * @details
	 *  ~ 该类用于从相机采集cv::Mat格式的对象，采集到图像后会立即将图像上传到显存中。
	 *  ~ 也就是说，能从该采集器处获取cv::Mat和cv::cuda::GpuMat对象，
	 *    但需要注意，二者均对应同一张图像，故当任意一者被获取后，另一者就不再是“最新的”，即“latest”。
	 */
	class GpuMatAcquisitor : public MatAcquisitor
	{
	protected:
		/// 显存图像矩阵互斥量
		mutable std::shared_mutex GpuPictureMutex {};
		/// 位于显存中的图像矩阵对象
		cv::cuda::GpuMat GpuPicture {};

	public:
		/// 构造函数
		explicit GpuMatAcquisitor(CameraDevice* device) : MatAcquisitor(device)
		{}

		/**
		 * @brief 接收到图片事件
		 * @param data 原始图片数据
		 * @details
		 *  ~ 当采集到图片并处理完毕时，该方法将被采集线程调用。
		 */
		void ReceivePictureIncomeEvent(AbstractAcquisitor::RawPicture data) override;

		/**
		 * @brief 获取图片
		 * @param wait_for_latest 是否阻塞当前线程直到采集到新的图片，若为false，则函数将直接返回，无论图片是否是最新的
		 * @return 采集到的图片
		 * @throw std::logic_error 当设备未开始采集时调用该方法将抛出该异常
		 * @throw std::runtime_error 当设备开始采集但却异常离线时调用方法将抛出该异常
		 */
		cv::cuda::GpuMat GetGpuPicture(bool wait_for_latest = false) noexcept(false);
	};

}

#endif
