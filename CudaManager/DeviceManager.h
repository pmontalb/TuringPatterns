#pragma once

#include "Common.h"
#include "DeviceManagerHelper.h"

namespace dev
{
	// This class is exported from the CudaManager.dll
	class CUDAMANAGER_API CDeviceManager 
	{
	public:
		static CDeviceManager& Get();

		size_t GetDeviceCount();

		void SetDevice(size_t i);
		void SetBestDevice();
		void CheckDeviceSanity();

	private:
		CDeviceManager();
		~CDeviceManager() = default;

		// Avoid copying this singleton
		CDeviceManager(const CDeviceManager&) = delete;
		CDeviceManager(CDeviceManager&&) = delete;
		CDeviceManager& operator=(const CDeviceManager&) = delete;
		CDeviceManager& operator=(CDeviceManager&&) = delete;

		static CDeviceManager* instance;
	};
}

