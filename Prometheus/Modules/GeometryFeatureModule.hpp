#pragma once

#include <tuple>
#include <unordered_set>
#include <opencv4/opencv2/opencv.hpp>

namespace RoboPioneers::Modules
{
	/**
	 * @brief 几何特征模块
	 * @author Vincent
	 * @details
	 *  ~ 该静态类提供几何特征相关方法。
	 */
	class GeometryFeatureModule
	{
	public:
		/// 标准化的几何信息
		struct GeometryFeature
		{
			/**
			 * @brief 匹配形状
			 * @details
			 *  ~ 匹配形状可以表明几何参数的匹配来源。
			 *  ~ 默认值为Unknown，即未知匹配类型。
			 *  ~ 几何特征模块的几何特征计算方法会自动设定为相应的形状。
			 */
			enum class Shape
			{
				/// 未知匹配类型
				Unknown,
				/// 椭圆匹配
				Ellipse,
				/// 矩形匹配
				Rectangle
			}MatchingShape {Shape::Unknown};

			/// 原始信息
			struct {
				/// 轮廓
				std::vector<cv::Point> Contour;
				/// 没有标准化的外接矩形
				cv::RotatedRect CircumscribedRectangle;
			}Raw;

			/**
			 * @brief 转角
			 * @details
			 *  ~ 标准化朝向定义为矩形长边对应的方向，范围为[0,180)。
			 *  ~ 转角为X轴沿逆时针旋转后与长边平行时所转过的角度的绝对值。
			 *  ~ 若旋转矩形为正方形，则确定能使得转角在[0,45)范围的转角为长边。
			 */
			double Angle {0.0f};
			/**
			 * @brief 宽度
			 * @details
			 *  ~ 宽度为矩形的短边对应的长度。
			 */
			double Width {0.0f};
			/**
			 * @brief 长度
			 * @details
			 *  ~ 长度为矩形的长边对应的长度。
			 */
			double Length {0.0f};

			/**
			 * @brief 向量信息
			 * @details
			 *  ~ 存储旋转矩形的相关向量。
			 */
			struct
			{
				/**
				 * @brief 方向向量
				 * @details
				 *  ~ 为一个宽的中心指向另一个宽的中心的向量的一半，方向沿朝向。
				 */
				cv::Vec2f Direction;
				/**
				 * @brief 顺时针半对角线向量
				 * @details
				 *  ~ 从矩形中心点指向方向向量顺时针旋转后第一个与之重合的角点的向量。
				 */
				cv::Vec2f ClockwiseDiagonal;
				/**
				 * @brief 逆时针半对角线向量
				 * @details
				 *  ~ 从矩形中心点指向方向向量顺时针旋转后第一个与之重合的角点的向量。
				 */
				cv::Vec2f AnticlockwiseDiagonal;
			}Vectors;

			/// 中心坐标
			cv::Point2i Center;
		};

		/**
		 * @brief 获取旋转矩形的几何参数
		 * @param rotated_rectangle 旋转矩形对象
		 * @param contour 边界信息
		 * @return 标准化旋转矩形参数
		 * @struct GeometryFeatureModule::GeometryFeature
		 * @details
		 *  ~ 标准化的处理标准需要查看GeometryFeature的注释。
		 */
		static auto StandardizeRotatedRectangle(const cv::RotatedRect& rotated_rectangle,
										        const std::vector<cv::Point>& contour = std::vector<cv::Point>())
			-> GeometryFeature;

		/**
		 * @brief 获取矩形的几何特征
		 * @param contour 边缘点集
		 * @return 外接点集的最小旋转矩形的几何特征
		 * @struct GeometryFeatureModule::GeometryFeature
		 * @details
		 *  ~ 标准化的处理标准需要查看GeometryFeature的注释。
		 */
		static auto GetRectangleGeometryFeature(const std::vector<cv::Point>& contour)
			-> GeometryFeature;

		/**
		 * @brief 获取椭圆的几何特征
		 * @param contour 边缘点集
		 * @return 外接点集的最小旋转椭圆的几何特征
		 * @struct GeometryFeatureModule::GeometryFeature
		 * @details
		 *  ~ 标准化的处理标准需要查看GeometryFeature的注释。
		 */
		static auto GetEllipseGeometryFeature(const std::vector<cv::Point>& contour)
			-> GeometryFeature;

		/**
		 * @brief 比较两个几何特征是否相等
		 * @param a 一个几何特征
		 * @param b 另一个几何特征
		 * @retval true 当几何特征相同
		 * @retval false 当几何特征不同
		 * @details
		 *  ~ 注意，只比较标准化信息，不比较原始信息。
		 */
		static bool IsGeometryFeatureIdentical(const GeometryFeature& a, const GeometryFeature& b);

		using ElementPair = std::tuple<GeometryFeature, GeometryFeature>;

		/// 元素对哈希函数
		struct ElementPairHash
		{
			std::size_t operator()(const ElementPair& elementPair) const
			{
//				auto center = elementPair.A.Center + elementPair.B.Center;
				auto center = std::get<0>(elementPair).Center + std::get<1>(elementPair).Center;
				/// 零点到中心点的距离平方
				auto square_distance = center.dot(center);

				return std::hash<decltype(square_distance)>()(square_distance);
			}
		};

		/// 元素对相等判断函数
		struct ElementPairEqual
		{
			bool operator()(const ElementPair& a, const ElementPair& b) const
			{
				if (Modules::GeometryFeatureModule::IsGeometryFeatureIdentical(std::get<0>(a), std::get<0>(b)) &&
				    Modules::GeometryFeatureModule::IsGeometryFeatureIdentical(std::get<1>(a), std::get<1>(b)))
				{
					return true;
				}
				if (Modules::GeometryFeatureModule::IsGeometryFeatureIdentical(std::get<0>(a), std::get<1>(b)) &&
				    Modules::GeometryFeatureModule::IsGeometryFeatureIdentical(std::get<1>(a), std::get<0>(b)))
				{
					return true;
				}

				return false;
			}
		};

		using ElementPairSet = std::unordered_set<ElementPair, ElementPairHash, ElementPairEqual>;
	};
}