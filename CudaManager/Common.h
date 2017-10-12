#pragma once

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the CUDAMANAGER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// CUDAMANAGER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef CUDAMANAGER_EXPORTS
	#define CUDAMANAGER_API __declspec(dllexport)
#else
	#define CUDAMANAGER_API __declspec(dllimport)
#endif

//extern CUDAMANAGER_API int nCudaManager;
//CUDAMANAGER_API int fnCudaManager(void);

//source
//// This is an example of an exported variable
//CUDAMANAGER_API int nCudaManager = 0;
//
//// This is an example of an exported function.
//CUDAMANAGER_API int fnCudaManager(void)
//{
//	return 42;
//}