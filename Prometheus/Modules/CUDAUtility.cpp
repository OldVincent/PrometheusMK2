#include "CUDAUtility.hpp"

extern void CUDADeviceSynchronize();

namespace RoboPioneers::Modules
{
	/// 同步设备对象
	void CUDAUtility::SynchronizeDevice()
	{
		CUDADeviceSynchronize();
	}
}
