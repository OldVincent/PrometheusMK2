#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <cmath>
#include <type_traits>

namespace RoboPioneers::Modules
{
	/**
	 * @brief 数学实用工具模块
	 * @author Vincent
	 * @details
	 *  ~ 该模块封装了常用的数学方法。
	 */
	class MathUtility
	{
	public:
		/**
		 * @brief 判断值是否在范围内
		 * @tparam StrictMode 是否启用严格模式，具体查阅细节部分
		 * @tparam ValueType 值类型
		 * @param value 目标值
		 * @param lower_bound 小边界
		 * @param upper_bound 大边界
		 * @return 若在边界内，则返回true，否则返回false
		 * @details
		 *  ~ 启用严格模式前，判断值是否在[lower_bound, upper_bound]内。
		 *  ~ 启用严格模式后，判断值是否在(lower_bound, upper_bound)内。
		 */
		template<bool StrictMode = false, typename ValueType>
		static inline bool InRange(ValueType value, ValueType lower_bound, ValueType upper_bound)
		{
			if constexpr (!StrictMode)
			{
				if (value >= lower_bound && value <= upper_bound)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
				// 此处需要else，因为if-constexpr会影响编译时的代码结构
			else
			{
				if (value > lower_bound && value < upper_bound)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
		}

		/**
		 * @brief 求正交向量
		 * @param direction 原方向向量
		 * @return 正交向量
		 */
		static cv::Vec2f OrthogonalVector(const cv::Vec2f &direction);

		/**
		 * @brief 求两条直线的交点
		 * @param line1point 线1上的任一点
		 * @param line1direction 与线1平行的向量
		 * @param line2point 线2上的任一点
		 * @param line2direction 与线2平行的向量
		 * @return 交点坐标
		 */
		static auto LineCrossPoint(const cv::Point2f &line1point, const cv::Vec2f &line1direction,
							 const cv::Point2f &line2point, const cv::Vec2f &line2direction)
							 -> std::optional<cv::Point2f>;

		/**
		 * @brief 求向量a与的b夹角的cos
		 * @param a 向量
		 * @param b 向量
		 * @return 夹角的余弦值
		 */
		static double CosIncludedAngle(cv::Vec2f a, cv::Vec2f b);

		/**
		 * @brief 将二维点转换为二维向量
		 * @param point 点
		 * @return 二维向量
		 */
		template<typename FromType, typename ToType>
		static inline cv::Vec<ToType, 2> CastToVector(const cv::Point_<FromType>& point)
		{
			return cv::Vec<ToType, 2>{point.x, point.y};
		}

		/**
		 * @brief 将二维点转换为二维向量
		 * @param point 点
		 * @return 二维向量
		 */
		template<typename FromType, typename ToType>
		static inline cv::Point_<ToType> CastToPoint(const cv::Vec<FromType, 2>& vector)
		{
			return cv::Point_<ToType>{vector(0), vector(1)};
		}

		/**
		 * @brief 点加向量，返回点
		 * @tparam PointType 点内数据存储类型
		 * @tparam VectorType 向量内数据存储类型
		 * @param point 点
		 * @param vector 向量
		 * @return 点
		 */
		template<typename PointType, typename VectorType>
		static inline cv::Point_<PointType> PointAddVector(cv::Point_<PointType> point, cv::Vec<VectorType, 2> vector)
		{
			return cv::Point_<PointType>{
				point.x + static_cast<PointType>(vector(0)),
				point.y + static_cast<PointType>(vector(1))
			};
		}

		/**
		 * @brief 向量加点，返回向量
		 * @tparam PointType 点内数据存储类型
		 * @tparam VectorType 向量内数据存储类型
		 * @param point 点
		 * @param vector 向量
		 * @return 向量
		 */
		template<typename VectorType, typename PointType>
		static inline cv::Vec<VectorType, 2> VectorAddPointer(cv::Vec<VectorType, 2> vector, cv::Point_<PointType> point)
		{
			return cv::Point_<PointType>{
					static_cast<VectorType>(point.x) + vector(0),
					static_cast<VectorType>(point.y) + vector(1)
			};
		}

		/**
		 * @brief 变更点内存储数据的类型
		 * @tparam ToType 目标类型
		 * @tparam FromType 源类型
		 * @param point 点
		 * @return 变更类型后的点
		 */
		template<typename ToType, typename FromType>
		static inline cv::Point_<ToType> ChangePointType(cv::Point_<FromType> point)
		{
			return cv::Point_<ToType>{
					static_cast<ToType>(point.x),
					static_cast<ToType>(point.y)
			};
		}

		/**
		 * @brief 变更向量内存储数据的类型
		 * @tparam ToType 目标类型
		 * @tparam FromType 源类型
		 * @param vector 向量
		 * @return 变更类型后的向量
		 */
		template<typename ToType, typename FromType>
		static inline cv::Vec<ToType, 2> ChangeVectorType(cv::Vec<FromType, 2> vector)
		{
			return cv::Vec<ToType, 2>{
					static_cast<ToType>(vector(0)),
					static_cast<ToType>(vector(1))
			};
		}

		/**
		 * @brief 相似系数
		 * @tparam ValueType 值的类型
		 * @param value1 参与比较的值
		 * @param value2 参与比较的值
		 * @return 范围-1.0f~1.0f
		 * @details
		 *  ~ 相似系数即绝对值较小的那一方的值除以绝对值较大的那一方的值。
		 */
		template<typename ValueType1, typename ValueType2>
		static inline double ResembleCoefficient(ValueType1 value1, ValueType2 value2)
		{
			if (value1 == 0) value1 = 0.00001;
			if (value2 == 0) value2 = 0.00001;

			ValueType1 abs_value1 = value1 >= 0 ? value1 : -value1;
			ValueType2 abs_value2 = value2 >= 0 ? value2 : -value2;

			return abs_value1 <= abs_value2 ? static_cast<double>(value1) / static_cast<double>(value2) :
			       static_cast<double>(value2) / static_cast<double>(value1);
		}

		/**
		 * @brief 最小值限制
		 * @tparam ValueType 值类型
		 * @param target 目标值
		 * @param min 最小值
		 * @return 限制后的值 >= min
		 */
		template<typename ValueType>
		static inline ValueType RestrictByMin(ValueType target, ValueType min)
		{
			if (target < min)
				target = min;
			return target;
		}

		/**
		 * @brief 最大值限制
		 * @tparam ValueType 值类型
		 * @param target 目标值
		 * @param max 最大值
		 * @return 限制后的值 <= max
		 */
		template<typename ValueType>
		static inline ValueType RestrictByMax(ValueType target, ValueType max)
		{
			if (target > max)
				target = max;
			return target;
		}

		/**
		 * @brief 将值限制在范围内
		 * @tparam ValueType 值类型
		 * @param target 要被限制的值
		 * @param min 最小值
		 * @param max 最大值
		 * @return 限制后的结果 [min, max]
		 */
		template<typename ValueType>
		static inline ValueType RestrictInRange(ValueType target, ValueType min, ValueType max)
		{
			if (target < min)
				target = min;
			else if (target > max)
				target = max;
			return target;
		}

		/**
		 * @brief 不带限制地中心固定缩放矩形
		 * @param target 目标矩形
		 * @param scale 缩放比例，0.0f~1.0f
		 * @return 缩放后的矩形
		 */
		static cv::Rect ScaleRectangle(const cv::Rect& target, cv::Vec2d scale);

		/**
		 * @brief 带限制地中心固定缩放矩形
		 * @param target 目标矩形
		 * @param scale 缩放比例
		 * @param max_size
		 * @return
		 */
		static cv::Rect ScaleRectangle(const cv::Rect& target, cv::Vec2d scale, const cv::Size& max_size);
	};
}
