#include "ArmorMatchingService.hpp"

#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

#include <tbb/tbb.h>
#include <shared_mutex>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "../Modules/CUDAUtility.hpp"
#include "../Modules/ImageDebugUtility.hpp"
#include "../Modules/MathUtility.hpp"

namespace RoboPioneers::Prometheus
{
	//==============================
	// 控制方法
	//==============================

	#ifdef DEBUG
	cv::Mat DebugPictureForArmor;
	std::shared_mutex DebugPictureForArmorMutex;
	#endif

	/// 更新方法
	void ArmorMatchingService::OnUpdate(Sparrow::Frame &frame)
	{
		#ifdef DBUEG
		DebugPictureForArmor = frame.CutPicture.clone();
		#endif

		Output.PossibleArmors.clear();

		std::list<std::list<GeometryFeature> const *> pairs_groups {Input.PossibleRectangles, Input.PossibleEllipses};

		for (auto const * pairs : pairs_groups)
		{
			auto group_result = SearchPossiblePairs(*pairs);
			Output.PossibleArmors.insert(group_result.begin(), group_result.end());
		}

		#ifdef DEBUG
			cv::Mat debug_picture = frame.CutPicture.clone();
			for (const auto& passed_pair : Output.PossibleArmors)
			{
				cv::line(debug_picture, std::get<0>(passed_pair).Center + *Input.CuttingAreaPositionOffset,
			 std::get<1>(passed_pair).Center, cv::Scalar(127, 255, 0), 3);
			}
			cv::imshow("Target", debug_picture);

		#endif
	}

	/// 搜索可能的灯条对
	auto ArmorMatchingService::SearchPossiblePairs(
			const std::list<GeometryFeature> &elements) -> ElementPairSet
	{
		ElementPairSet result;
		std::shared_mutex result_mutex;

		ElementPairSet checked_paris;
		std::shared_mutex checked_pairs_mutex;

		tbb::parallel_for_each(elements.begin(), elements.end(), [
				this, &elements,
				&result, &result_mutex, &checked_paris, &checked_pairs_mutex
				](const GeometryFeature& target){
			auto&& [success_pairs, failed_pairs] = this->MatchPairs(target, elements,
														   checked_paris, checked_pairs_mutex);

			std::unique_lock result_lock(result_mutex);
			result.insert(success_pairs.begin(), success_pairs.end());
			result_lock.unlock();

			std::unique_lock checked_pairs_lock(checked_pairs_mutex);
			checked_paris.insert(success_pairs.begin(), success_pairs.end());
			checked_paris.insert(failed_pairs.begin(), failed_pairs.end());
			checked_pairs_lock.unlock();
		});

		return result;
	}

	/// 进行匹配
	auto ArmorMatchingService::MatchPairs(const ArmorMatchingService::GeometryFeature &target,
	                                      const std::list<GeometryFeature> &elements,
	                                      const ArmorMatchingService::ElementPairSet &ignored_pairs,
	                                      std::shared_mutex &ignored_pairs_mutex) -> MatchingResult
	{
		MatchingResult result;
		std::shared_mutex result_mutex;

		tbb::parallel_for_each(elements.begin(), elements.end(), [
				this, &target, &ignored_pairs, &ignored_pairs_mutex,
				&result, &result_mutex](const GeometryFeature& candidate){
			//==============================
			// 检查是否需要进行匹配判断
			//==============================

			if (&candidate == &target) return;
			std::shared_lock ignored_pairs_lock(ignored_pairs_mutex);
			if (ignored_pairs.find(ElementPair{target, candidate}) != ignored_pairs.end()) return;
			ignored_pairs_lock.unlock();

			//==============================
			// 调用函数进行判断
			//==============================

			std::unique_lock result_lock(result_mutex);
			if (this->CheckGeometryConditions(target, candidate))
			{
				result.PassedPairs.emplace(ElementPair{target, candidate});
			}
			else
			{
				result.FailedPairs.emplace(ElementPair{target, candidate});
			}
		});

		return result;
	}

	/// 核验几何学条件
	bool ArmorMatchingService::CheckGeometryConditions(const ArmorMatchingService::GeometryFeature &a,
	                                                   const ArmorMatchingService::GeometryFeature &b) const
	{
		/// 根据形状绑定需要使用的设定信息
		const SettingsType::ShapeSettings& settings = a.MatchingShape == GeometryFeature::Shape::Rectangle ?
				Settings.RectangleSettings : Settings.EllipseSettings;

		//==============================
		// 内八和外八剔除
		//==============================
		if (Modules::MathUtility::ResembleCoefficient(a.Angle, 90.0f) < settings.IgnorantDifferentDirectionRatio &&
			Modules::MathUtility::ResembleCoefficient(b.Angle, 90.0f) < settings.IgnorantDifferentDirectionRatio)
		{
			if ((a.Angle - 90.0f) * (b.Angle - 90.0f) < 0)
			{
				return false;
			}
		}

		//==============================
		// 高度比检测
		//==============================
		double height_ratio = std::fmin(a.Length, b.Length) / std::fmax(a.Length, b.Length);
		if (height_ratio < settings.MinHeightRatio)
		{
			return false;
		}

		//==============================
		// 角度比检测
		//==============================

		double a_stable_angle = a.Raw.Contour.size() > 5 ? Modules::GeometryFeatureModule::GetEllipseGeometryFeature(a.Raw.Contour).Angle : a.Angle;
		double b_stable_angle = b.Raw.Contour.size() > 5 ? Modules::GeometryFeatureModule::GetEllipseGeometryFeature(b.Raw.Contour).Angle : b.Angle;

		double angle_ratio = std::fmin(a_stable_angle, b_stable_angle) / std::fmax(a_stable_angle, b_stable_angle);
		if (angle_ratio < settings.MinAngleRatio)
		{
			return false;
		}

		//==============================
		// 连线垂直于其中一边
		//==============================

		constexpr double pi = 3.14159265;
		constexpr double half_pi = pi / 2;

		auto raw_link_direction = a.Center - b.Center;
		auto link_direction = cv::Vec2f(raw_link_direction.x, raw_link_direction.y);

		// a垂直于连线的程度
		double a_perpendicular_ratio = std::acos(
				Modules::MathUtility::CosIncludedAngle(a.Vectors.Direction, link_direction)
		) / half_pi;

		// b垂直于连线的程度
		double b_perpendicular_ratio =  std::acos(
				Modules::MathUtility::CosIncludedAngle(b.Vectors.Direction, link_direction)
		) / half_pi;

		if (a_perpendicular_ratio < settings.MinLinkPerpendicularRatio &&
		b_perpendicular_ratio < settings.MinLinkPerpendicularRatio)
		{
			return false;
		}

		//==============================
		// 至少其中一个元素的垂直线能与另一个元素相交
		//==============================

		// a方向的正交向量
		auto a_orthogonal = Modules::MathUtility::OrthogonalVector(a.Vectors.Direction);
		// 从a引出的正交线与b中心线的交点
		auto a_cross_point = Modules::MathUtility::LineCrossPoint(a.Center, a_orthogonal,
															b.Center, b.Vectors.Direction);
		if (!a_cross_point)
		{
			return false;
		}

		std::vector<cv::Point2f> b_vertices;
		b_vertices.resize(4);
		b.Raw.CircumscribedRectangle.points(b_vertices.data());
		auto a_intersected_distance = - cv::pointPolygonTest(b.Raw.Contour, *a_cross_point, true);
		bool a_intersected = a_intersected_distance <= b.Length * settings.CrossPointMaxDistanceRatio;

		// b方向的正交向量
		auto b_orthogonal = Modules::MathUtility::OrthogonalVector(b.Vectors.Direction);
		// 从b引出的正交线与b中心线的交点
		auto b_cross_point = Modules::MathUtility::LineCrossPoint(b.Center, b_orthogonal,
		                                                          a.Center, a.Vectors.Direction);
		if (!b_cross_point)
		{
			return false;
		}
		std::vector<cv::Point2f> a_vertices;
		a_vertices.resize(4);
		a.Raw.CircumscribedRectangle.points(a_vertices.data());
		double b_intersected_distance = - cv::pointPolygonTest(a.Raw.Contour, *b_cross_point, true);
		bool b_intersected = b_intersected_distance <= a.Length * settings.CrossPointMaxDistanceRatio;

		#ifdef DEBUG
			std::shared_lock lock(DebugPictureForArmorMutex);
//		    cv::putText(DebugPictureForArmor, std::to_string(a_intersected_distance).substr(0, 4),
//		                *a_cross_point + cv::Point2f(5,-5), cv::FONT_HERSHEY_COMPLEX, 0.3,
//		                cv::Scalar(255, 255, 255));
//		    cv::putText(DebugPictureForArmor, std::to_string(b_intersected_distance).substr(0, 4),
//		                *b_cross_point + cv::Point2f(5,-5), cv::FONT_HERSHEY_COMPLEX, 0.3,
//		                cv::Scalar(255, 255, 255));
//		    cv::putText(DebugPictureForArmor, std::to_string(b.Length).substr(0, 4),
//		                b.Center + cv::Point2i(5,-5), cv::FONT_HERSHEY_COMPLEX, 0.3,
//		                cv::Scalar(255, 0, 255));
//		    cv::putText(DebugPictureForArmor, std::to_string(b.Length).substr(0, 4),
//		                a.Center + cv::Point2i(5,-5), cv::FONT_HERSHEY_COMPLEX, 0.3,
//		                cv::Scalar(255, 0, 255));
			cv::circle(DebugPictureForArmor, *a_cross_point, 3, cv::Scalar(205, 90, 106), 2);
			cv::circle(DebugPictureForArmor, *b_cross_point, 3, cv::Scalar(205, 90, 106), 2);
			lock.unlock();
		#endif

		if (!(b_intersected && a_intersected))
		{
			return false;
		}

		return true;
	}

}