#pragma once

#include <SerialPortDriver/SerialPortDriver.hpp>
#include <CameraDriver/CameraDriver.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <stdexcept>

namespace RoboPioneers::Sparrow
{
	/// 应用类
	class Application;

	/**
	 * @brief 运行时类
	 * @author Vincent
	 * @details
	 *  ~ 该类用于维护引擎的单实例模块。
	 *  ~ 该类工作在单例模式下。
	 */
	class Runtime
	{
	private:
		//==============================
		// 私有设定与状态
		//==============================

		/**
		 * @brief 设定文件路径
		 * @details
		 *  ~ 该路径为设定文件的路径。
		 *  ~ 若未给定或找到设定文件，则为空。
		 */
		std::string SettingFilePathSource;

		/**
		 * @brief 程序配置
		 * @details
		 *  ~ 对应用于存储配置的JSON文件。
		 */
		boost::property_tree::ptree ProgramSettings;

		/**
		 * @brief 程序命令行选项
		 * @details
		 *  ~ 对应在执行程序的命令后给出的参数。
		 */
		boost::program_options::variables_map ProgramOptions;

	public:
		//==============================
		// 公开设定与状态
		//==============================

		/**
		 * @brief 设定文件路径
		 * @details
		 *  ~ 该路径为设定文件的路径。
		 *  ~ 若未给定或找到设定文件，则为空。
		 */
		const decltype(SettingFilePathSource)& SettingFilePath {SettingFilePathSource};

		//==============================
		// 静态控制方法
		//==============================

		/**
		 * @brief 获取实例
		 * @return 唯一实例，保证非空
		 */
		static Runtime* GetInstance() noexcept;

		//==============================
		// 生命周期控制方法
		//==============================

		/**
		 * @brief 启动应用
		 * @param arguments_count 参数个数
		 * @param arguments 参数字符串数组的指针
		 * @param application 要启动的应用对象指针
		 * @details
		 *  ~ 后续将开启应用的生命周期。
		 *  ~ 给定的参数用于辅助确定配置文件的位置。
		 */
		void Launch(int arguments_count, char** arguments, Application* application);

		//==============================
		// 基本方法
		//==============================

		template<typename Type>
		Type ReadOptions(const std::string& name)
		{
			if (!ProgramSettings.empty())
			{
				return ProgramSettings.get<Type>(name);
			}
			else
			{
				throw std::runtime_error("Runtime::ReadSetting Setting JSON File Is Not Given.");
			}
		}

		/**
		 * @brief 读取设定
		 * @tparam Type 类型
		 * @param name 设定的 名称
		 * @throw std::runtime_error 当未给定设定文件路径且未查找到设定文件
		 * @return 设定的值
		 */
		template<typename Type>
		Type ReadSetting(const std::string& name)
		{
			if (!ProgramSettings.empty())
			{
				return ProgramSettings.get<Type>(name);
			}
			else
			{
				throw std::runtime_error("Runtime::ReadSetting Setting JSON File Is Not Given.");
			}
		}
	};

	/**
	 * @brief 引擎对象指针
	 * @details
	 *  ~ 该指针在构造时指针来自运行时的获取实例方法。
	 *  ~ 注意，在C++17以下的版本，inline关键字无法保证其只初始化一次，故将失效。
	 */
	inline Runtime* const Engine = Runtime::GetInstance();
}