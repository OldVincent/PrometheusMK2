#pragma once

#include <CameraDriver/CameraDriver.hpp>

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief HSV双采集器
	 * @author Vincent
	 * @details
	 *  ~ 该类其他功能均与普通的采集器一致，只不过其会在采集线程中将图像转化为HSV格式。
	 */
	class HSVDualMatAcquisitor : public Modules::CameraDriver::Acquisitors::DualMatAcquisitor
	{
	public:
		/// 构造函数
		HSVDualMatAcquisitor(Modules::CameraDriver::CameraDevice* camera) :
		Modules::CameraDriver::Acquisitors::DualMatAcquisitor(camera)
		{}

		/// 接受到原始图片
		void ReceivePictureIncomeEvent(AbstractAcquisitor::RawPicture data) override;
	};
}