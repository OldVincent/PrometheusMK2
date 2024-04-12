#include "TargetEncodeService.hpp"

#include "../Modules/CRCModule.hpp"

namespace RoboPioneers::Prometheus
{
	/// 更新方法
	void TargetEncodeService::OnUpdate(Sparrow::Frame &frame)
	{
		Output.Data.clear();
		Output.Data.resize(11);

		*reinterpret_cast<char*>(&Output.Data[0]) = *Input.Command;
		*reinterpret_cast<int*>(&Output.Data[1]) = *Input.X;
		*reinterpret_cast<int*>(&Output.Data[5]) = *Input.Y;
		*reinterpret_cast<char*>(&Output.Data[9]) = *Input.Number;
		Output.Data[10] = Modules::CRCModule::GetCRC8CheckSum(Output.Data.data(), 10);
	}
}