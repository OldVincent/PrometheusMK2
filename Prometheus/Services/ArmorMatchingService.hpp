#pragma once

#include <SparrowEngine/SparrowEngine.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <unordered_set>
#include <tuple>

#include "../Modules/GeometryFeatureModule.hpp"

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 索敌服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于根据输入的HSV图片搜索装甲板，并将可能的结果输出。
	 */
	class ArmorMatchingService : public Sparrow::Service
	{
	public:
		//==============================
		// 中间态结构
		//==============================

		/// 几何特征结构体
		using GeometryFeature = Modules::GeometryFeatureModule::GeometryFeature;
		/// 元素对
		using ElementPair = Modules::GeometryFeatureModule::ElementPair;
		/// 元素对集合
		using ElementPairSet = Modules::GeometryFeatureModule::ElementPairSet;


		//==============================
		// 输入输出部分
		//==============================

		/// 输入结构体
		struct {
			/// 可能的矩形
			std::list<GeometryFeature> const * PossibleRectangles;
			/// 可能的椭圆
			std::list<GeometryFeature> const * PossibleEllipses;

			cv::Point* CuttingAreaPositionOffset;
		}Input;

		/// 输出结构体
		struct {
			/// 结果
			ElementPairSet PossibleArmors;
		}Output;

		//==============================
		// 设定部分
		//==============================

		/**
		 * @brief 选项结构体
		 * @details
		 *  ~ 该结构体存储的值只影响到单个服务实例。
		 */
		struct SettingsType{
			/**
			 * @brief 形状设定
			 * @details
			 *  ~ 该结构体存储针对不同的形状的不同的设定值。
			 *  ~ 结构体的默认值是为矩形匹配元素设定的。
			 */
			struct ShapeSettings
			{
				/// 多大程度接近垂直时忽略转角不同向的问题
				double IgnorantDifferentDirectionRatio {0.6f};

				/**
			 	 * @brief 最大相交距离，用于正交线检测，大小为0.0f到1.0f
			 	 * @details
			 	 *  ~ 该条件可以有效削弱零散噪点的影响。
			 	 *  ~ 减少该值以进一步削弱零散噪点影响。
			 	 */
				double CrossPointMaxDistanceRatio {0.8f};

				/**
			 	 * @brief 最小高度比，值为0.0f到0.1f
			 	 * @details
			 	 *  ~ 高度比表示两个灯条高度相似程度。
			 	 */
				double MinHeightRatio {0.7};

				/// 最小角度比，值为0.0f到0.1f
				/**
				 * @brief 最小角度比，值为0.0f到0.1f
				 * @detail
				 *  ~ 角度比表示两个待匹配元素在多大程度上近似平行。
				 *  ~ 提高该阈值可以减少多灯条匹配错误的几率。
				 */
				double MinAngleRatio {0.7f};

				/// 连线最小垂直程度
				double MinLinkPerpendicularRatio {0.3f};

			}RectangleSettings;

			/// 椭圆特化的设定，要求比较宽松
			ShapeSettings EllipseSettings{
				.IgnorantDifferentDirectionRatio = 0.6f,
				.CrossPointMaxDistanceRatio = 0.8f,
				.MinHeightRatio = 0.7f,
				.MinAngleRatio = 0.7f,
				.MinLinkPerpendicularRatio = 0.3f
			};
		}Settings;

		/// 元素对
//		struct ElementPair {
//			GeometryFeature A;
//			GeometryFeature B;
//		};

		/**
		 * @brief 搜索可能的元素对
		 * @param element 元素列表
		 * @return 可能的元素对列表
		 */
		[[nodiscard]] auto SearchPossiblePairs(const std::list<GeometryFeature>& elements) -> ElementPairSet;

		/// 匹配结果
		struct MatchingResult
		{
			/// 通过检测的元素对
			ElementPairSet PassedPairs;
			/// 未通过检测的元素对
			ElementPairSet FailedPairs;
		};

		/**
		 * @brief 为目标元素匹配可能的元素对
		 * @param target 目标
		 * @param ignored_pairs 忽略的目标
		 * @return 匹配的元素结果，第一个元素为成功匹配的元素对，第二个元素为失败的元素对
		 */
		[[nodiscard]] auto MatchPairs(const GeometryFeature& target, const std::list<GeometryFeature>& elements,
								const ElementPairSet &ignored_pairs, std::shared_mutex &ignored_pairs_mutex)
			-> MatchingResult;

		/**
		 * @brief 核验元素对
		 * @param a 一个元素
		 * @param b 另一个元素
		 * @retval true 当元素对满足条件
		 * @retval false 当元素对不满足条件
		 */
		[[nodiscard]] bool CheckGeometryConditions(const GeometryFeature& a, const GeometryFeature& b) const;

	protected:
		/// 更新方法
		void OnUpdate(Sparrow::Frame &frame) override;
	};
}
