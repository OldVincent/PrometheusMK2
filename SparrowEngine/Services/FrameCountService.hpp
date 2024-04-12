#pragma once

#include "../Framework/Service.hpp"

#include <chrono>

namespace RoboPioneers::Sparrow
{
	/**
	 * @brief 帧统计服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于计算帧速率。
	 *  ~ 其计算算法为统计总共多少帧所需要的时间会超过设定的统计周期的时间。
	 */
	class FrameCountService : public Service
	{
	private:
		/// 周期开始时间
		std::chrono::time_point<std::chrono::steady_clock> PeriodBeginTime;
		/// 帧数
		std::size_t CurrentCount {0};
		/// 当前时间跨度
		std::chrono::milliseconds CurrentPeriodSpan {1000};

		/// 帧速率
		std::size_t FrameRateSource {0};
		/// 上一周期帧数
		std::size_t LastPeriodFrameCountSource {0};

	public:
		/// 构造函数
		FrameCountService();

		/**
		 * @brief 周期时间跨度，单位为毫秒
		 * @details
		 *  ~ 周期时间即统计时间，若设定为500毫秒，期间统计了20帧，则每秒帧速率会被计算为40帧。
		 *  ~ 设定越大则更新越频繁。
		 *  ~ 变更将在当前周期结束后生效。
		 */
		 std::chrono::milliseconds PeriodSpan {CurrentPeriodSpan};

		 /**
		  * @brief 是否自动输出帧率
		  * @details
		  *  ~ 若为true，则会在统计周期结束后自动在std::cout流中输出帧速率。
		  */
		 bool AutoPrint {true};

		 /**
		  * @brief 帧速率
		  * @details
		  *  ~ 该速率为上个周期计算的速率。
		  */
		 const decltype(FrameRateSource)& FrameRate {FrameRateSource};

		 /**
		  * @brief 上一周期帧数
		  * @details
		  *  ~ 该个数为上一个统计周期统计的总帧数。
		  */
		 const decltype(LastPeriodFrameCountSource)& LastPeriodFrameCount {LastPeriodFrameCountSource};

	protected:
		/// 更新事件
		void OnUpdate(Frame &frame) override;
	};
}