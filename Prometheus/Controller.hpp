#pragma once

#include <SparrowEngine/SparrowEngine.hpp>

#include "Services/ColorPerceptionService.hpp"
#include "Services/LightBarSearchingService.hpp"
#include "Services/ArmorMatchingService.hpp"
#include "Services/BattleIntelligenceService.hpp"
#include "Services/TargetEncodeService.hpp"
#include "Services/KeyTerminationService.hpp"
#include "Services/PictureCuttingService.hpp"

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 控制器类
	 * @author Vincent
	 * @details
	 *  ~ 该类用于控制自动索敌系统的生命周期与执行逻辑。
	 */
	class Controller : public Sparrow::Application
	{
	public:
		/// 构造函数
		Controller();

	protected:
		/// 服务结构体
		struct {
			/// 图像裁剪单元
			PictureCuttingService PictureCuttingUnit;
			/// 颜色感知单元
			ColorPerceptionService ColorPerceptionUnit;
			/// 灯条搜索单元
			LightBarSearchingService LightBarSearchingUnit;
			/// 装甲板匹配单元
			ArmorMatchingService ArmorMatchingUnit;
			/// 战斗智能单元
			BattleIntelligenceService BattleIntelligenceUnit;
			/// 目标数据包编码单元
			TargetEncodeService TargetEncodeUnit;
			/// 按键终止单元
			KeyTerminationService KeyTerminationUnit;
		}Services;

	protected:
		/**
		 * @brief 更新方法
		 * @param frame 帧对象
		 * @return 需要写入下位机的字节
		 */
		std::vector<unsigned char> OnUpdate(Sparrow::Frame &frame) override;

		/// 配置设备
		void OnConfigureDevices() override;

		/// 安装服务
		void OnInstallServices() override;

		/// 卸载服务
		void OnUninstallServices() override;
	};
}