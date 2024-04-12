#pragma once

#include <SparrowEngine/SparrowEngine.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <list>

#include "../Modules/GeometryFeatureModule.hpp"

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 颜色感知服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于从画面中检测出在敌人颜色范围内的区域。
	 */
	class ColorPerceptionService : public Sparrow::Service
	{
	public:
		//==============================
		// 输入输出部分
		//==============================

		/// 输出
		struct {
			/// 感知蒙版图片，在范围内的像素点的位置被设置为255，否则为0
			cv::cuda::GpuMat MaskPicture;
		}Output;

		//==============================
		// 设定部分
		//==============================

		/// 目标颜色类型
		enum class TargetColorEnum {
			/// 目标为红色
			Red,
			/// 目标为蓝色
			Blue
		};

		/// 颜色感知设定
		struct {
			/// 目标颜色
			TargetColorEnum TargetColor {TargetColorEnum::Red};

			/// 红色系处理设定
			struct {
				/// 色调第一段最大值 0~Hue1UpperBound
				int Hue1UpperBound {55};
				/// 色调第二段最大值 Hue2LowerBound~180
				int Hue2LowerBound {180};
				/// 饱和度最小值 SaturationLowerBound~255
				int SaturationLowerBound {0};
				/// 亮度最小值 ValueLowerBound~255
				int ValueLowerBound {180};
			}RedThresholds;

			/// 蓝色系处理设定
			struct {
				/// 色调第二段最大值 HueLowerBound~HueUpperBound
				int HueLowerBound {78};
				/// 色调第一段最大值 HueLowerBound~HueUpperBound
				int HueUpperBound {155};
				/// 饱和度最小值 SaturationLowerBound~255
				int SaturationLowerBound {0};
				/// 亮度最小值 ValueLowerBound~255
				int ValueLowerBound {46};
			}BlueThresholds;
		}Settings;

		/**
		 * @brief 资产结构体
		 * @details
		 *  ~ 该结构体存储可以复用的、一般贯穿服务生命周期的对象。
		 */
		struct {
			/**
			 * @brief 单通道闭运算滤波器
			 * @details
			 *  ~ 采用十字形状的滤波器效果更好，比矩形优化效果显著。
			 */
			cv::Ptr<cv::cuda::Filter> CloseFilter {
					cv::cuda::createMorphologyFilter(
							cv::MORPH_CLOSE, CV_8UC1,
							cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5)))
			};
		}Properties;

	protected:

		//==============================
		// 处理过程
		//==============================

		/**
		 * @brief 筛选红色色系区域
		 * @param hsv_picture HSV色域上的彩色图片
		 * @return 目标区域蒙版
		 */
		[[nodiscard]] cv::cuda::GpuMat FilterRedArea(const cv::cuda::GpuMat& hsv_picture) const;

		/**
		 * @brief 筛选蓝色色系区域
		 * @param hsv_picture HSV色域上的彩色图片
		 * @return 目标区域蒙版
		 */
		[[nodiscard]] cv::cuda::GpuMat FilterBlueArea(const cv::cuda::GpuMat& hsv_picture) const;

		/// 更新方法
		void OnUpdate(Sparrow::Frame &frame) override;
	};
}