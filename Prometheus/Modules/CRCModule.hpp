#pragma once

namespace RoboPioneers::Modules
{
	/**
	 * @brief CRC校验静态模块
	 * @author Vincent
	 * @details
	 *  ~ 该模块提供CRC相关静态方法。
	 */
	class CRCModule
	{
	public:
		/**
		 * @brief 获取8位CRC校验码
		 * @param data 数据指针
		 * @param length 数据长度
		 * @return 8位CRC校验码
		 */
		static unsigned char GetCRC8CheckSum(unsigned char* data, unsigned int length);

		/**
		 * @brief 获取16位CRC校验码
		 * @param data 数据指针
		 * @param length 数据长度
		 * @return 16位CRC校验码
		 */
		static unsigned short GetCRC16CheckSum(unsigned char* data, unsigned int length);
	};
}