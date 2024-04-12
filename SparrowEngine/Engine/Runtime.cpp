#include "Runtime.hpp"

#include <boost/filesystem.hpp>

#include "../Framework/Application.hpp"

namespace RoboPioneers::Sparrow
{
	/// 获取运行时对象的实例
	Runtime* Runtime::GetInstance() noexcept
	{
		static Runtime runtime_instance;
		return &runtime_instance;
	}

	/// 启动应用
	void Runtime::Launch(int arguments_count, char** arguments, Application* application)
	{
		//==============================
		// 处理配置信息
		//==============================

		// 处理程序参数
		{
			boost::program_options::options_description options("Options:");
			options.add_options()
					("setting,s", boost::program_options::value<std::string>(&SettingFilePathSource),
					        "Use Settings in a Setting File")
					("debug,d", boost::program_options::value<bool>(),
					        "Enable the Debug Mode or Not");
			// 解析参数
			boost::program_options::store(
					boost::program_options::parse_command_line(arguments_count, arguments, options),
					ProgramOptions);
			// 通知并修改绑定的外部变量
			boost::program_options::notify(ProgramOptions);
		}
		// 处理配置文件
		{
			if (!SettingFilePath.empty() && boost::filesystem::exists(SettingFilePath))
			{
				// 解析设定文件
				boost::property_tree::json_parser::read_json(SettingFilePath, ProgramSettings);
			}
		}

		//==============================
		// 启动应用
		//==============================

		// 开启生命旗标
		application->InnerSettings.LifeFlag = true;

		// 安装设备
		application->Install();

		// 生命周期循环
		while(application->InnerSettings.LifeFlag)
		{
			application->Update();
		}

		// 卸载设备
		application->Uninstall();
	}
}