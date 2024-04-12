#include "MathUtility.hpp"

#include <cmath>

namespace RoboPioneers::Modules
{
	/// 求线交点
	auto MathUtility::LineCrossPoint(const cv::Point2f &line1point, const cv::Vec2f &line1direction,
	                                        const cv::Point2f &line2point, const cv::Vec2f &line2direction)
		-> std::optional<cv::Point2f>
	{
		if (line1direction(1) == 0 || line2direction(1) == 0)
		{
			return std::nullopt;
		}

		// 线1的斜率
		double k1 = line1direction(1) / line1direction(0);
		// 线2的斜率
		double k2 = line2direction(1) / line2direction(0);

		// 平行则不计算交点
		if (std::fabs(k1 - k2) < 0.001)
		{
			return std::nullopt;
		}

		double x = (k1 * line1point.x - k2 * line2point.x) / (k1 -  k2) - (line1point.y - line2point.y) / (k1 - k2);
		double y = k1 * (x - line1point.x) + line1point.y;

		return cv::Point2f(x, y);
	}

	/// 求正交向量
	cv::Vec2f MathUtility::OrthogonalVector(const cv::Vec2f &direction)
	{
		return cv::Vec2f(-direction(1), direction(0));
	}

	/// 计算夹角余弦
	double MathUtility::CosIncludedAngle(const cv::Vec2f a, const cv::Vec2f b)
	{
		return a.ddot(b) / cv::norm(a) / cv::norm(b);
	}

	/// 缩放矩形
	cv::Rect MathUtility::ScaleRectangle(const cv::Rect& target, cv::Vec2d scale)
	{
		cv::Rect result;

		double delta_width = target.width * (scale(0) - 1);
		double delta_height = target.height * (scale(1) - 1);
		result.x = target.x - static_cast<int>(delta_width / 2);
		result.y = target.y - static_cast<int>(delta_height / 2);
		result.width = target.width + delta_width;
		result.height = target.height + delta_height;

		return result;
	}

	cv::Rect MathUtility::ScaleRectangle(const cv::Rect& target, cv::Vec2d scale, const cv::Size& max_size)
	{
		cv::Rect result;

		double delta_width = target.width * (scale(0) - 1);
		double delta_height = target.height * (scale(1) - 1);
		result.x = RestrictInRange(target.x - static_cast<int>(delta_width / 2), 0, max_size.width);
		result.y = RestrictInRange(target.y - static_cast<int>(delta_height / 2), 0, max_size.height);
		result.width = RestrictInRange(target.width + static_cast<int>(delta_width), 0, max_size.width - result.x);
		result.height = RestrictInRange(target.height + static_cast<int>(delta_height), 0, max_size.height - result.y);

		return result;
	}
}