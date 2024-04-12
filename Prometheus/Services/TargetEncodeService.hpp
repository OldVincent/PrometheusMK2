#pragma once

#include <SparrowEngine/SparrowEngine.hpp>
#include <vector>

namespace RoboPioneers::Prometheus
{
	/**
	 * @brief 目标数据包编码服务
	 * @author Vincent
	 * @details
	 *  ~ 该服务用于将目标数据编码成2021版上下位机通信数据包。
	 */
	class TargetEncodeService : public Sparrow::Service
	{
	public:
		//==============================
		// 输入部分
		//==============================

		struct {
			char const *Command;
			int const *X;
			int const *Y;
			char const *Number;
		}Input;

		//==============================
		// 输出部分
		//==============================

		struct {
			/**
		 	 * @brief 编码数据
		 	 * @details
		 	 *  ~ 将中心点横纵坐标编码为数据包。
		 	 *  ~ 包长10个字节。
		 	 *  ~ 0~3为float横坐标，4~7为float纵坐标，8为bool是否找到，9为CRC8位校验码。
		 	 */
			std::vector<unsigned char> Data;
		}Output;

	protected:
		/// 更新方法
		void OnUpdate(Sparrow::Frame &frame) override;
	};
}