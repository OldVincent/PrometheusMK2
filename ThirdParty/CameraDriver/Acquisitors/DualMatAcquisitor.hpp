#pragma  once

#ifndef NO_CUDA

#include "MatAcquisitor.hpp"

namespace RoboPioneers::Modules::CameraDriver::Acquisitors
{
	/**
	 * @brief 双图片获取器
	 * @author Vincent
	 * @details
	 *  ~ 该类的存在是为了解决GpuMatAcquisitor中GpuPicture和Picture共享IsLatest属性的问题。
	 */
	class DualMatAcquisitor : public MatAcquisitor
	{
	public:
		/**
		 * @brief 构造函数
		 * @param device 相机设备指针
		 */
		explicit DualMatAcquisitor(CameraDevice* device) : MatAcquisitor(device)
		{}

	protected:
		/// 显存中的图片
		cv::cuda::GpuMat GpuPicture {};

	public:
		//==============================
		// 采集器基本控制方法
		//==============================

		/**
		 * @brief 获取图片
		 * @param wait_for_latest 是否阻塞当前线程直到采集到新的图片，若为false，则函数将直接返回，无论图片是否是最新的
		 * @return 元组，第一个元素为内存中的该图片，第二个元素为显存中的该图片
		 * @throw std::logic_error 当设备未开始采集时调用该方法将抛出该异常
		 * @throw std::runtime_error 当设备开始采集但却异常离线时调用方法将抛出该异常
		 */
		std::tuple<cv::Mat, cv::cuda::GpuMat> GetDualPicture(bool wait_for_latest);

		//==============================
		// 事件处理方法
		//==============================

		/**
		 * @brief 接收到图片事件
		 * @param data 原始图片数据
		 * @details
		 *  ~ 当采集到图片并处理完毕时，该方法将被采集线程调用。
		 */
		void ReceivePictureIncomeEvent(AbstractAcquisitor::RawPicture data) override;
	};
}

#endif
