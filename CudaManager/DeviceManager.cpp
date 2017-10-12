// CudaManager.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "DeviceManager.h"
#include "DeviceManagerHelper.h"

namespace dev
{
	CDeviceManager* CDeviceManager::instance = nullptr;

	CDeviceManager& CDeviceManager::Get()
	{
		if (!instance)
			instance = new CDeviceManager();

		return *instance;
	}

	CDeviceManager::CDeviceManager()
	{
		SetBestDevice();
	}


	void CDeviceManager::SetDevice(size_t i)
	{
		detail::SetDevice(i);
		CheckDeviceSanity();
	}

	void CDeviceManager::SetBestDevice()
	{
		int bestDevice = -1;
		detail::GetBestDevice(bestDevice);
		SetDevice(bestDevice);
	}

	void CDeviceManager::CheckDeviceSanity()
	{	
		detail::GetDeviceStatus();
	}

	size_t CDeviceManager::GetDeviceCount()
	{
		int ret;
		detail::GetDeviceCount(ret);

		return ret;
	}
}
