#pragma once

#include <opencv4/opencv2/opencv.hpp>

namespace RoboPioneers::Modules
{
	/**
	 * @brief 图像处理实用工具
	 * @author Vincent
	 * @details
	 *  ~ 该模块提供了图像处理常用方法。
	 */
	class ImageDebugUtility
	{
	public:
		/**
		 * @brief 绘制矩形
		 * @param canvas 画布，将被画图形的地方
		 * @param center 矩形的中心位置
		 * @param size 矩形的尺寸
		 * @param color 矩形边框的颜色
		 * @param thickness 矩形边框的厚度
		 */
		static void DrawRectangle(cv::Mat& canvas,
							const cv::Point& center, const cv::Size& size,
							const cv::Scalar& color, int thickness = 3);

		/**
		 * @brief 绘制旋转矩形
		 * @param canvas 画布，将被画图形的地方
		 * @param center 矩形的中心位置
		 * @param size 矩形的尺寸
		 * @param angle 矩形的旋转角度
		 * @param color 矩形边框的颜色
		 * @param thickness 矩形边框的厚度
		 */
		static void DrawRotatedRectangle(cv::Mat& canvas,
								   const cv::Point& center, const cv::Size& size, float angle,
								   const cv::Scalar& color, int thickness = 3);

		/**
		 * @brief 绘制旋转矩形
		 * @param canvas 画布，将被画图形的地方
		 * @param rotated_rect 旋转矩形对象
		 * @param color 矩形边框的颜色
		 * @param thickness 矩形边框的厚度
		 */
		static void DrawRotatedRectangle(cv::Mat& canvas,
		                                 const cv::RotatedRect& rotated_rect,
		                                 const cv::Scalar& color, int thickness = 3);

		/**
		 * @brief 显示GPU图片
		 * @param title 窗口标题
		 * @param picture 显示的图片
		 */
		static void ShowGPUPicture(const std::string& title, const cv::cuda::GpuMat& picture);
	};
}
