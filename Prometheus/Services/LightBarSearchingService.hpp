#pragma once

#include <SparrowEngine/SparrowEngine.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <list>

#include "../Modules/GeometryFeatureModule.hpp"

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 灯条搜索单元
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于从颜色蒙版中搜索灯条。
	 */
	class LightBarSearchingService : public Sparrow::Service
	{
	private:
		/// 标准化旋转矩形
		using GeometryFeature = Modules::GeometryFeatureModule::GeometryFeature;

	public:
		/// 输入
		struct {
			/// 颜色蒙版
			cv::cuda::GpuMat* BinaryPicture{nullptr};
		}Input;

		/// 输出
		struct {
			/// 可能的矩形
			std::list<GeometryFeature> PossibleRectangles;
			/// 可能的椭圆
			std::list<GeometryFeature> PossibleEllipses;
		}Output;

		struct {
			/// 椭圆填充率阈值
			double EllipseFillingRateThreshold {0.6};
			/// 矩形填充率阈值
			double RectangleFillingRateThreshold {0.6};
		}Settings;

	protected:
		/// 更新事件
		void OnUpdate(Sparrow::Frame &frame) override;

		/// 支持的匹配形状
		enum class Shape
		{
			/// 该元素被按照矩形匹配
			Rectangle,
			/// 该元素被按照椭圆匹配
			Ellipse
		};

		/// 可能的元素
		struct PossibleElement
		{
			/// 标准化后的几何参数
			GeometryFeature GeometryParameters;
			/// 匹配的形状
			Shape MatchingShape {Shape::Rectangle};

			/// 置信度，0.0f ~ 1.0f
			double Confidence {0.0f};
		};

		/// 搜索结果
		struct SearchingResult {
			/// 矩形匹配形状的几何特征列表
			std::list<GeometryFeature> Rectangles;
			/// 椭圆匹配形状的几何特征列表
			std::list<GeometryFeature> Ellipses;
		};

		/**
		 * @brief 搜索可能的图形元素
		 * @param binary_picture 目标二值图
		 * @return 搜索结果，第一个成员为匹配矩形的列表，第二个成员为匹配椭圆的列表
		 */
		[[nodiscard]] auto SearchPossibleElements(const cv::Mat& binary_picture) const
			-> SearchingResult;

		/**
		 * @brief 检验几何学特征
		 * @param contour 目标轮廓
		 * @retval true 当目标几何参数满足条件
		 * @retval false 当目标几何参数不满足条件
		 */
		[[nodiscard]] auto CheckGeometryConditions(const std::vector<cv::Point> &contour) const
			-> std::optional<PossibleElement>;
	};
}