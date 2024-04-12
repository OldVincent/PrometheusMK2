#include "BattleIntelligenceService.hpp"

#include "../Modules/ImageDebugUtility.hpp"
#include "../Modules/MathUtility.hpp"
#include "../Modules/GeometryFeatureModule.hpp"
#include <iostream>

namespace RoboPioneers::Prometheus
{
	#ifdef DEBUG
	cv::Mat DebugPictureIntelligence;
	#endif

	void BattleIntelligenceService::OnUpdate(Sparrow::Frame &frame)
	{
		#ifdef DEBUG
		DebugPictureIntelligence = frame.CutPicture.clone();
		#endif

		Output.InterestedArea = cv::Rect(0,0,0,0);
		Output.NeedToCut = false;
		bool found = false;
		cv::RotatedRect best_one;
		double best_one_score = 0;
		ElementPair best_pair;

		if (!Input.PossibleArmors->empty())
		{
			for (const auto& candidate : *Input.PossibleArmors)
			{
				auto current_rectangle = CastPairToRotatedRectangle(candidate);
				auto geometry_parameters = Modules::GeometryFeatureModule::StandardizeRotatedRectangle(current_rectangle);

				if (geometry_parameters.Angle > 45.0f && geometry_parameters.Angle < 135.0f)
				{
					break;
				}

				constexpr double standard_small_armor_aspect_ratio = 23.5f / 6.0f;
				constexpr double standard_big_armor_aspect_ratio = 14.0f / 6.0f;

				double small_armor_ratio = Modules::MathUtility::ResembleCoefficient(
						geometry_parameters.Width / geometry_parameters.Length, standard_small_armor_aspect_ratio);

				double big_armor_ratio = Modules::MathUtility::ResembleCoefficient(
						geometry_parameters.Width / geometry_parameters.Length, standard_big_armor_aspect_ratio
				);

				double score = (small_armor_ratio > big_armor_ratio ? small_armor_ratio : big_armor_ratio) *
				               Modules::MathUtility::ResembleCoefficient(geometry_parameters.Angle, 90) *
				               geometry_parameters.Length * geometry_parameters.Width;

				if (score >= best_one_score)
				{
					best_pair = candidate;
					best_one = current_rectangle;
					found = true;
				}
			}
		}

		if (found)
		{
			cv::Rect interested_area = best_one.boundingRect();

			if (interested_area.width == 0 || interested_area.height == 0)
			{
				found = false;
			}
			else
			{
				interested_area.x = interested_area.x + frame.PointOffset.x;
				interested_area.y = interested_area.y + frame.PointOffset.y;
				interested_area = Modules::MathUtility::ScaleRectangle(interested_area, {4.0f, 6.0f},
														   cv::Size(1280, 1024));

				TrackingInterestedArea = interested_area;

				TrackingRemainTimes = Settings.TrackingFrames;
				Tracked = true;

				Output.Command = CommandSet::Fire;
				Output.X = (std::get<0>(best_pair).Center.x + std::get<1>(best_pair).Center.x ) / 2 + frame.PointOffset.x;
				Output.Y = (std::get<0>(best_pair).Center.y + std::get<1>(best_pair).Center.y ) / 2 + frame.PointOffset.y;


				std::cout << "Found: X:" << Output.X << " Y:" << Output.Y << std::endl;
			}
		}
		if (!found)
		{
			Output.Command = CommandSet::Standby;

			TrackingRemainTimes--;

			if (Tracked)
			{
				Modules::MathUtility::ScaleRectangle(TrackingInterestedArea, {8.0f, 12.0f},
										 cv::Size(1280, 1024));
			}
		}

		if (Tracked)
		{
			Output.InterestedArea = TrackingInterestedArea;
			if (!Output.InterestedArea.empty())
			{
				Output.NeedToCut = true;
			}
		}

		if (TrackingRemainTimes <= 0)
		{
			Tracked = false;
		}

		#ifdef DEBUG
			best_one.center = best_one.center + Modules::MathUtility::ChangePointType<float>(frame.PointOffset);
			Modules::ImageDebugUtility::DrawRotatedRectangle(DebugPictureIntelligence, best_one, cv::Scalar(0,255,255));
			cv::imshow("Final Decision", DebugPictureIntelligence);
		#endif
	}

	/// 从灯条对匹配旋转矩形
	cv::RotatedRect BattleIntelligenceService::CastPairToRotatedRectangle(
			const BattleIntelligenceService::ElementPair &pair)
	{
		std::vector<cv::Point> mixed_contour;
		mixed_contour.reserve(std::get<0>(pair).Raw.Contour.size() + std::get<1>(pair).Raw.Contour.size());
		mixed_contour.insert(mixed_contour.end(), std::get<0>(pair).Raw.Contour.begin(), std::get<0>(pair).Raw.Contour.end());
		mixed_contour.insert(mixed_contour.end(), std::get<1>(pair).Raw.Contour.begin(), std::get<1>(pair).Raw.Contour.end());
		return cv::minAreaRect(mixed_contour);
	}

	/// 判断是否是同一块装甲板
	bool BattleIntelligenceService::IsSameArmor(const cv::RotatedRect& current_target, const cv::RotatedRect& previous_target) const
	{
		std::vector<cv::Point> intersection_region {};
		auto intersection_type = cv::rotatedRectangleIntersection(current_target, previous_target, intersection_region);
		switch (intersection_type)
		{
			default:
			case cv::RectanglesIntersectTypes::INTERSECT_NONE:
				return false;
				break;
			case cv::RectanglesIntersectTypes::INTERSECT_FULL:
				return true;
			case cv::RectanglesIntersectTypes::INTERSECT_PARTIAL:
				auto intersection_area = cv::contourArea(intersection_region);
				// 要求重叠面积占比达标
				if (intersection_area / previous_target.size.area() < Settings.IntersectionAreaRatioThreshold)
					return false;
				// 要求角度近乎平行
				auto geometry_current_target =
						Modules::GeometryFeatureModule::StandardizeRotatedRectangle(current_target);
				auto geometry_previous_target =
						Modules::GeometryFeatureModule::StandardizeRotatedRectangle(previous_target);
				if (Modules::MathUtility::ResembleCoefficient(geometry_current_target.Angle, geometry_previous_target.Angle)
				< Settings.AngleRatioThreshold)
					return false;

				return true;;
		}
	}

	//==============================
	// 状态机
	//==============================

	//==============================
	// 搜索状态
	//==============================

	bool BattleIntelligenceService::UpdateSearchStatus(ElementPairSet *possible_candidates)
	{
		std::list<cv::RotatedRect> candidates_to_search;

		for (const auto& current_candidate : *possible_candidates)
		{
			auto current_rectangle = CastPairToRotatedRectangle(current_candidate);

			bool current_matched = false;

			for (auto& [previous_rectangle, data] : Candidates)
			{
				if (IsSameArmor(current_rectangle, previous_rectangle))
				{
					current_matched = true;

					data.AppearanceCount += 1;
					data.RemainFrames -= 1;

					break;
				}
			}

			if (!current_matched)
			{
				Candidates.emplace_back(current_rectangle, CandidateStatus{});
			}
		}

		for (auto iterator = Candidates.begin(); iterator != Candidates.end();)
		{
			if (std::get<1>(*iterator).RemainFrames <= 0)
			{
				if (std::get<1>(*iterator).AppearanceCount >= Settings.AppearanceThreshold)
				{
					candidates_to_search.emplace_back(std::get<0>(*iterator));
				}

				iterator = Candidates.erase(iterator);
			}
			else
			{
				++iterator;
			}
		}

		#ifdef DEBUG
		for (const auto& result : candidates_to_search)
		{
			Modules::ImageDebugUtility::DrawRotatedRectangle(DebugPictureIntelligence, result, cv::Scalar(205,0,0), 3);
		}
		#endif

		#ifdef DEBUG
			cv::imshow("Battle Intelligence", DebugPictureIntelligence);
		#endif

		return false;
	}


}