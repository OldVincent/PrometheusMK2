#include "GeometryFeatureModule.hpp"

#include <cmath>

namespace RoboPioneers::Modules
{
	/// 获取旋转矩形标准信息
	auto GeometryFeatureModule::StandardizeRotatedRectangle(
			const cv::RotatedRect &rotated_rectangle,
			const std::vector<cv::Point>& contour) -> GeometryFeatureModule::GeometryFeature
	{
		//==============================
		// 注意
		// 角度的范围为-0到-180，一定包含-0度，
		// 但不确定是否包含-180度
		//==============================

		// 标准化参数结果
		GeometryFeature parameters;

		//==============================
		// 填充信息
		//==============================

		parameters.Center = rotated_rectangle.center;
		parameters.Raw.CircumscribedRectangle = rotated_rectangle;
		parameters.Raw.Contour = contour;

		//==============================
		// 确定标准化长宽、转角和向量
		//==============================

		cv::Point2f vertices[4];
		rotated_rectangle.points(vertices);

		const double angle = rotated_rectangle.angle >= 0.0f ? rotated_rectangle.angle : -rotated_rectangle.angle;

		if (rotated_rectangle.size.width >= rotated_rectangle.size.height)
		{
			//==============================
			// 宽大于等于高
			//==============================

			parameters.Length = rotated_rectangle.size.width;
			parameters.Width = rotated_rectangle.size.height;

			if (rotated_rectangle.angle <= 0)
			{
				// 角度为逆时针
				if (angle < 180.0f)
				{
					// [0, -180)

					parameters.Angle = angle;

					parameters.Vectors.Direction = vertices[2] - vertices[1];
					parameters.Vectors.ClockwiseDiagonal = vertices[3] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[2] - rotated_rectangle.center;
				}
				else
				{
					// 角度等于180 [180, 180]

					parameters.Angle = angle;

					parameters.Vectors.Direction = vertices[1] - vertices[2];
					parameters.Vectors.ClockwiseDiagonal = vertices[1] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[0] - rotated_rectangle.center;
				}
			}
			else
			{
				// (0, 180]

				parameters.Angle = 180 - angle;

				parameters.Vectors.Direction = vertices[1]  - vertices[2];
				parameters.Vectors.ClockwiseDiagonal = vertices[1] - rotated_rectangle.center;
				parameters.Vectors.AnticlockwiseDiagonal = vertices[0] - rotated_rectangle.center;
			}
		}
		else
		{
			//==============================
			// 高大于宽
			//==============================

			parameters.Length = rotated_rectangle.size.height;
			parameters.Width = rotated_rectangle.size.width;

			if (rotated_rectangle.angle <= 0)
			{
				// 角度为逆时针

				if (angle < 90.0f)
				{
					// [0, 90)

					parameters.Angle = 90.0f + angle;

					parameters.Vectors.Direction = vertices[1] - vertices[0];
					parameters.Vectors.ClockwiseDiagonal = vertices[2] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[1] - rotated_rectangle.center;
				}
				else
				{
					// [90 ~ 180]

					parameters.Angle = angle - 90.0f;

					parameters.Vectors.Direction = vertices[0] - vertices[1];
					parameters.Vectors.ClockwiseDiagonal = vertices[0] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[3] - rotated_rectangle.center;
				}
			}
			else
			{
				// 角度为顺时针

				if (rotated_rectangle.angle <= 90.0f)
				{
					parameters.Angle = 90.0f - angle;

					parameters.Vectors.Direction = vertices[1] - vertices[0];
					parameters.Vectors.ClockwiseDiagonal = vertices[2] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[1] - rotated_rectangle.center;
				}
				else
				{
					parameters.Angle = 270.0f - angle;

					parameters.Vectors.Direction = vertices[0] - vertices[1];
					parameters.Vectors.ClockwiseDiagonal = vertices[0] - rotated_rectangle.center;
					parameters.Vectors.AnticlockwiseDiagonal = vertices[3] - rotated_rectangle.center;
				}
			}
		}
		return parameters;
	}

	/// 获取矩形几何特征
	auto GeometryFeatureModule::GetRectangleGeometryFeature(const std::vector<cv::Point> &contour)
		-> GeometryFeatureModule::GeometryFeature
	{
		auto result = StandardizeRotatedRectangle(cv::minAreaRect(contour));
		result.Raw.Contour = contour;
		result.MatchingShape = GeometryFeature::Shape::Rectangle;
		return result;
	}

	/// 获取椭圆几何特征
	auto GeometryFeatureModule::GetEllipseGeometryFeature(const std::vector<cv::Point> &contour)
		-> GeometryFeatureModule::GeometryFeature
	{
		auto result = StandardizeRotatedRectangle(cv::fitEllipseDirect(contour));
		result.Raw.Contour = contour;
		result.MatchingShape = GeometryFeature::Shape::Ellipse;
		return result;
	}

	/// 比较几何特征
	bool GeometryFeatureModule::IsGeometryFeatureIdentical(const GeometryFeatureModule::GeometryFeature &a,
	                                                       const GeometryFeatureModule::GeometryFeature &b)
	{
		if (a.Center.x != b.Center.x || a.Center.y != b.Center.y) return false;
		if (a.Width != b.Width || a.Length != b.Length) return false;
		if (a.Raw.CircumscribedRectangle.angle != b.Raw.CircumscribedRectangle.angle) return false;
		if (a.Angle != b.Angle) return false;
		if (a.Raw.Contour != b.Raw.Contour) return false;
		return true;
	}
}
