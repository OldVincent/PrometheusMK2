#pragma once

#include <SparrowEngine/SparrowEngine.hpp>
#include <unordered_set>
#include <shared_mutex>
#include <list>
#include <tuple>
#include "../Modules/GeometryFeatureModule.hpp"

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 战斗智能服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于从可能的装甲板中选择一个并推荐。
	 */
	class BattleIntelligenceService : public Sparrow::Service
	{
	public:
		using GeometryFeature = Modules::GeometryFeatureModule::GeometryFeature;
		using ElementPair = Modules::GeometryFeatureModule::ElementPair;
		using ElementPairSet = Modules::GeometryFeatureModule::ElementPairSet;

		/// 元素对哈希函数
		struct ElementPairHash
		{
			std::size_t operator()(const ElementPair& elementPair) const
			{
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

		/// 输入
		struct {
			ElementPairSet* PossibleArmors;
		}Input;

		/// 输出
		struct {
			/// 指令字符
			char Command;
			/// 横坐标
			int X;
			/// 纵坐标
			int Y;
			/// 识别的数字
			char Number;

			/// 兴趣区域列表
			cv::Rect InterestedArea;
			/// 是否需要裁剪
			bool NeedToCut {false};
		}Output;

		/// 指令集
		struct CommandSet
		{
			/// 待机指令
			static constexpr char Standby = 0;
			/// 跟踪指令
			static constexpr char Track = 1;
			/// 开火指令
			static constexpr char Fire = 2;
		};

		struct {
			/// 追踪帧
			unsigned int TrackingFrames {5};

			/// 要求5帧内出现3帧
			unsigned int AppearanceThreshold {3};

			double InterestedAreaFirstScale {2.0};
			double InterestedAreaLostScale {3.0};

			/**
			 * @brief 重合面积阈值
			 * @details
			 *  ~ 当重合面积达到该比例后，将认为新旧装甲板是同一块装甲板。
			 */
			double IntersectionAreaRatioThreshold {0.5};
			/// 同装甲板角度近似系数
			double AngleRatioThreshold {0.5};
		}Settings;

	protected:
		/// 是否正在追踪
		bool Tracked = false;
		/// 跟踪区域
		cv::Rect TrackingInterestedArea;
		/// 跟踪剩余帧
		int TrackingRemainTimes {5};

		/// 更新方法
		void OnUpdate(Sparrow::Frame &frame) override;

		enum class StatusType
		{
			Search,
			Track
		}CurrentStatus {StatusType::Search};

		static cv::RotatedRect CastPairToRotatedRectangle(const ElementPair& pair);

		/**
		 * @brief 判断是否是同一块装甲板
		 * @param current_target 当前等待判断的目标旋转矩形
		 * @param previous_target 可能是原来的旋转矩形
		 * @retval true 当认为两个矩形是同一块装甲板
		 * @retval false 当认为两个矩形不是同一块装甲板
		 */
		[[nodiscard]] bool IsSameArmor(const cv::RotatedRect& current_target, const cv::RotatedRect& previous_target) const;

		//==============================
		// 搜索状态
		//==============================

		/// 目标状态
		struct CandidateStatus
		{
			/// 出现次数
			unsigned int AppearanceCount {1};
			/// 剩余检测帧数
			unsigned int RemainFrames {5};
		};

		/// 追踪候选列表
		std::list<std::tuple<cv::RotatedRect, CandidateStatus>> Candidates;

		bool UpdateSearchStatus(ElementPairSet* possible_candidates);

	};
}