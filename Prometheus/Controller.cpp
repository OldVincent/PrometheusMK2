#include "Controller.hpp"

#include <vector>

namespace RoboPioneers::Prometheus
{
	/// 构造方法
	Controller::Controller() : Sparrow::Application(0, "/dev/ttyTHS2")
	{
		InnerSettings.EnableSerialPort = true;

		InnerServices.FrameTimeController.Enable = false;
	}

	/// 更新方法
	std::vector<unsigned char> Controller::OnUpdate(Sparrow::Frame &frame)
	{

		Services.PictureCuttingUnit.Update(frame);
		Services.ColorPerceptionUnit.Update(frame);
		Services.LightBarSearchingUnit.Update(frame);
		Services.ArmorMatchingUnit.Update(frame);

		Services.BattleIntelligenceUnit.Update(frame);

		Services.TargetEncodeUnit.Update(frame);

		#ifdef DEBUG
		Services.KeyTerminationUnit.Update(frame);
		#endif

		return Services.TargetEncodeUnit.Output.Data;
	}

	/// 配置硬件
	void Controller::OnConfigureDevices()
	{
		/// 设置相机参数
		InnerDevices.Camera.SetExposureTime(1000);
		InnerDevices.Camera.SetGain(16);

		InnerDevices.Camera.SetWhiteBalance(Modules::CameraDriver::CameraDevice::WhiteBalanceChannel::Red, 1.5);
		InnerDevices.Camera.SetWhiteBalance(Modules::CameraDriver::CameraDevice::WhiteBalanceChannel::Blue, 1.3594);

		if (InnerSettings.EnableSerialPort)
		{
			InnerDevices.Port.SetBaudRate(115200);
			InnerDevices.Port.SetParityType(RoboPioneers::Modules::SerialPortDriver::SerialPort::parity::none);
			InnerDevices.Port.SetStopBitsType(RoboPioneers::Modules::SerialPortDriver::SerialPort::stop_bits::one);
			InnerDevices.Port.SetCharacterSize(8);
		}
	}

	/// 安装服务
	void Controller::OnInstallServices()
	{
		Services.PictureCuttingUnit.Input.CuttingArea = &Services.BattleIntelligenceUnit.Output.InterestedArea;
		Services.PictureCuttingUnit.Input.NeedToCut = &Services.BattleIntelligenceUnit.Output.NeedToCut;

		Services.LightBarSearchingUnit.Input.BinaryPicture = &Services.ColorPerceptionUnit.Output.MaskPicture;

		Services.ArmorMatchingUnit.Input.PossibleEllipses = &Services.LightBarSearchingUnit.Output.PossibleEllipses;
		Services.ArmorMatchingUnit.Input.PossibleRectangles = &Services.LightBarSearchingUnit.Output.PossibleRectangles;

		Services.BattleIntelligenceUnit.Input.PossibleArmors = &Services.ArmorMatchingUnit.Output.PossibleArmors;

		Services.TargetEncodeUnit.Input.Command = &Services.BattleIntelligenceUnit.Output.Command;
		Services.TargetEncodeUnit.Input.X = &Services.BattleIntelligenceUnit.Output.X;
		Services.TargetEncodeUnit.Input.Y = &Services.BattleIntelligenceUnit.Output.Y;
		Services.TargetEncodeUnit.Input.Number = &Services.BattleIntelligenceUnit.Output.Number;

		#ifdef DEBUG
		Services.KeyTerminationUnit.Enable = true;
		#else
		Services.KeyTerminationUnit.Enable = false;
		#endif
		Services.KeyTerminationUnit.Input.TargetLifeFlag = &InnerSettings.LifeFlag;
	}

	/// 卸载服务
	void Controller::OnUninstallServices()
	{

	}
}