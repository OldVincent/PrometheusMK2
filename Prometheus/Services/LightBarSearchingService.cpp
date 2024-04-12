#include "LightBarSearchingService.hpp"

#include <tbb/tbb.h>
#include <shared_mutex>
#include <optional>

#include "../Modules/GeometryFeatureModule.hpp"
#include "../Modules/MathUtility.hpp"
#include "../Modules/ImageDebugUtility.hpp"

namespace RoboPioneers::Prometheus
{
	#ifdef DEBUG
	cv::Mat DebugPictureForLightBar;
	std::shared_mutex DebugPictureForLightBarMutex;
	#endif

	/// 更新方法
	void LightBarSearchingService::OnUpdate(Sparrow::Frame &frame)
	{
		#ifdef DEBUG
		DebugPictureForLightBar = frame.OriginalPicture.clone();
		#endif

		cv::Mat binary_picture;
		Input.BinaryPicture->download(binary_picture);
		auto&& [rectangles, ellipses] = SearchPossibleElements(binary_picture);

		Output.PossibleRectangles = std::move(rectangles);
		Output.PossibleEllipses = std::move(ellipses);

		#ifdef DEBUG
		cv::imshow("Possible Light Bars", DebugPictureForLightBar);
		#endif
	}


	/// 搜索全部的矩形元素
	auto LightBarSearchingService::SearchPossibleElements(const cv::Mat& binary_picture) const -> SearchingResult
	{
		//==============================
		// 感知轮廓
		//==============================

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(binary_picture, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		//==============================
		// 遍历轮廓进行几何特征检测
		//==============================

		std::list<GeometryFeature> rectangle_result;
		std::shared_mutex rectangle_result_mutex;

		std::list<GeometryFeature> ellipse_result;
		std::shared_mutex ellipse_result_mutex;

		tbb::parallel_for_each(contours.begin(), contours.end(),[
				this, settings = &this->Settings,
				&rectangle_result, &rectangle_result_mutex,
				&ellipse_result, &ellipse_result_mutex]
				(const std::vector<cv::Point>& contour){
			auto&& element = this->CheckGeometryConditions(contour);

			if (element)
			{
				std::shared_mutex* mutex_pointer {nullptr};
				std::list<GeometryFeature>* result_pointer {nullptr};

				switch (element->MatchingShape)
				{
					case Shape::Ellipse:
						mutex_pointer = &ellipse_result_mutex;
						result_pointer = &ellipse_result;
						break;
					case Shape::Rectangle:
						mutex_pointer = &rectangle_result_mutex;
						result_pointer = &rectangle_result;
						break;
				}

				std::unique_lock lock(*mutex_pointer);
				result_pointer->push_back(element->GeometryParameters);
			}

			#ifdef DEBUG
				std::unique_lock lock(DebugPictureForLightBarMutex);

				Modules::ImageDebugUtility::DrawRotatedRectangle(
						DebugPictureForLightBar, element->GeometryParameters.Raw.CircumscribedRectangle,
						cv::Scalar(47,255,173), 1);
//				cv::line(DebugPictureForLightBar, element->GeometryParameters.Center,
//				         Modules::MathUtility::PointAddVector(element->GeometryParameters.Center,
//											   -element->GeometryParameters.Vectors.AnticlockwiseDiagonal),
//				         cv::Scalar(255, 0, 255), 3);
//				cv::line(DebugPictureForLightBar, element->GeometryParameters.Center,
//				         Modules::MathUtility::PointAddVector(element->GeometryParameters.Center,
//                                              -element->GeometryParameters.Vectors.ClockwiseDiagonal),
//				         cv::Scalar(255, 0, 255), 3);
//				cv::line(DebugPictureForLightBar, element->GeometryParameters.Center,
//				         Modules::MathUtility::PointAddVector(element->GeometryParameters.Center,
//				                                              element->GeometryParameters.Vectors.Direction),
//				         cv::Scalar(139, 139, 0), 3);
//				cv::putText(DebugPictureForLightBar,std::to_string(element->GeometryParameters.Raw.CircumscribedRectangle.angle).substr(0,4),
//					element->GeometryParameters.Center + cv::Point2i(15,-15),cv::FONT_HERSHEY_COMPLEX,0.3,
//					cv::Scalar(255, 255, 255));
//				cv::Point2f vertices[4];
//				element->GeometryParameters.Raw.CircumscribedRectangle.points(vertices);
//				cv::circle(DebugPictureForLightBar, vertices[0], 2, cv::Scalar(255,0,0),4);
//				cv::circle(DebugPictureForLightBar, vertices[1], 2, cv::Scalar(0,255,0),4);
//				cv::circle(DebugPictureForLightBar, vertices[2], 2, cv::Scalar(0,0,255),4);
				if (element->GeometryParameters.MatchingShape == GeometryFeature::Shape::Rectangle)
				{
					cv::circle(DebugPictureForLightBar, element->GeometryParameters.Center, 3, cv::Scalar(0, 0, 255), 3);
				}
				else
				{
					cv::circle(DebugPictureForLightBar, element->GeometryParameters.Center,3, cv::Scalar(255, 0, 0), 3);
				}
				lock.unlock();
			#endif
		});

		return SearchingResult{.Rectangles = rectangle_result, .Ellipses= ellipse_result};
	}

	/// 检验几何学特征
	auto LightBarSearchingService::CheckGeometryConditions(const std::vector<cv::Point> &contour)
		const -> std::optional<PossibleElement>
	{
		/// 矩形面积
		auto contour_area = cv::contourArea(contour);

		//==============================
		// 尝试匹配椭圆（旋转矩形面积填充率）
		//==============================

		if (contour.size() >= 5)
		{
			// 内接椭圆面积比，S=pi*a*b，a=0.5*width，b=0.5*height
			constexpr double InscribedEllipseAreaRatio = 3.1415926535f / 4.0f;

			auto ellipse_geometry_feature = Modules::GeometryFeatureModule::GetEllipseGeometryFeature(contour);
			auto matching_ellipse = ellipse_geometry_feature.Raw.CircumscribedRectangle;
			auto matching_ellipse_area = matching_ellipse.size.area() * InscribedEllipseAreaRatio;

			auto matching_ellipse_confidence = contour_area / matching_ellipse_area;

			if (matching_ellipse_area > 0 && matching_ellipse_confidence > Settings.EllipseFillingRateThreshold)
			{
				return PossibleElement{
						.GeometryParameters = ellipse_geometry_feature,
						.MatchingShape = Shape::Ellipse,
						.Confidence = matching_ellipse_confidence
				};
			}
		}

		//==============================
		// 尝试匹配矩形（旋转矩形面积填充率）
		//==============================

		auto rectangle_geometry_feature = Modules::GeometryFeatureModule::GetRectangleGeometryFeature(contour);
		auto matching_rectangle = rectangle_geometry_feature.Raw.CircumscribedRectangle;
		auto matching_rectangle_area = matching_rectangle.size.area();
		auto matching_rectangle_confidence = contour_area / matching_rectangle_area;

		if (matching_rectangle_area > 0 && matching_rectangle_confidence > Settings.RectangleFillingRateThreshold)
		{
			return PossibleElement{
					.GeometryParameters = rectangle_geometry_feature,
					.MatchingShape = Shape::Rectangle,
					.Confidence = matching_rectangle_confidence
			};
		}

		return std::nullopt;
	}
}