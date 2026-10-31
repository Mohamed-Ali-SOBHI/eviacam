/////////////////////////////////////////////////////////////////////////////
// Copyright:   (C) 2008-19 Cesar Mauri Loba - CREA Software Systems
// 
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
/////////////////////////////////////////////////////////////////////////////
#include "crvcamera_cv.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <sys/types.h>
#include <stdio.h>
#include <cassert>
#include <cstdlib>
#if defined(EVIACAM_MODERN_CAPTURE)
#ifdef WIN32
#include <dshow.h>
#elif defined(__APPLE__)
#include "../src/macos/platform.h"
#endif
#include <thread>
#include <chrono>
#endif

bool CCameraCV::g_cvInitialized= false;
int CCameraCV::g_numDevices= 0;
char CCameraCV::g_deviceNames[MAX_CV_DEVICES][50];

#if defined(WIN32)
#include <windows.h>

#define MAX_KEY_LENGTH 255
#define MAX_VALUE_NAME 16383

// Workround to enable capture at 30fps for some camera models
// Should be called with administrative rights
// Return 0 if ok, -1 if permission denied
int VfwCamFpsWorkaround ()
{
	HKEY hKey;    
	TCHAR    achKey[MAX_KEY_LENGTH];   // buffer for subkey name
    DWORD    cbName;                   // size of name string 
    TCHAR    achClass[MAX_PATH] = TEXT("");  // buffer for class name 
    DWORD    cchClassName = MAX_PATH;  // size of class string 
    DWORD    cSubKeys=0;               // number of subkeys 
    DWORD    cbMaxSubKey;              // longest subkey size 
    DWORD    cchMaxClass;              // longest class string 
    DWORD    cValues;              // number of values for key 
    DWORD    cchMaxValue;          // longest value name 
    DWORD    cbMaxValueData;       // longest value data 
    DWORD    cbSecurityDescriptor; // size of security descriptor 
    FILETIME ftLastWriteTime;      // last write time 
	DWORD i, retCode;   
   // DWORD cchValue = MAX_VALUE_NAME; 

	// Open key
	retCode= RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("SOFTWARE\\Microsoft\\VfWWDM Mapper"), 
							0, KEY_ALL_ACCESS, &hKey);
	if (retCode!= ERROR_SUCCESS) return -1;

	// Get the class name and the value count. 
    retCode = RegQueryInfoKey(
        hKey,                    // key handle 
        achClass,                // buffer for class name 
        &cchClassName,           // size of class string 
        NULL,                    // reserved 
        &cSubKeys,               // number of subkeys 
        &cbMaxSubKey,            // longest subkey size 
        &cchMaxClass,            // longest class string 
        &cValues,                // number of values for this key 
        &cchMaxValue,            // longest value name 
        &cbMaxValueData,         // longest value data 
        &cbSecurityDescriptor,   // security descriptor 
        &ftLastWriteTime);       // last write time 
 
	// Enumerate the subkeys
	if (cSubKeys) {
		for (i=0; i<cSubKeys; i++) { 
			cbName = MAX_KEY_LENGTH;
			retCode = RegEnumKeyEx(hKey, i,	achKey, &cbName, NULL, 
									NULL, NULL, &ftLastWriteTime); 

			if (retCode == ERROR_SUCCESS) {
				HKEY hKeyValue;

				retCode= RegOpenKeyEx(hKey, achKey, 0, KEY_WRITE, &hKeyValue);
				if (retCode== ERROR_SUCCESS) {
					DWORD value= 333330;
					retCode= RegSetValueEx(hKeyValue, TEXT("AvgTimePerFrame"), 0, REG_DWORD, 
											(const BYTE *) &value, sizeof(DWORD));
				}
				RegCloseKey(hKeyValue);
			}
		}
	} 

	// Close key
	RegCloseKey(hKey);

	return 0;
}

#endif

CCameraCV::CCameraCV(int cameraId, unsigned int width, int unsigned height, float fr)
{
	if (cameraId>= GetNumDevices()) throw camera_exception("wrong camera id");
	m_Id= cameraId;
	m_Width= width;
	m_Height= height;
	m_FrameRate= fr;
	m_pCvCapture = new cv::VideoCapture();

#if defined(WIN32) && !defined(EVIACAM_MODERN_CAPTURE)
	VfwCamFpsWorkaround ();
#endif
}

CCameraCV::~CCameraCV(void)
{
	Close ();	
	delete m_pCvCapture;
	m_pCvCapture = nullptr;
}

bool CCameraCV::DoOpen ()
{
	if (m_pCvCapture->isOpened()) return true;	// Already opened
#if defined(EVIACAM_MODERN_CAPTURE)
    const char* replay = std::getenv("EVIACAM_REPLAY");
    if (replay && *replay) {
        m_nextReplayTick = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        return m_pCvCapture->open(replay);
    }
#ifdef __APPLE__
    m_pCvCapture->open(m_Id, cv::CAP_AVFOUNDATION);
#else
    m_pCvCapture->open(m_Id, cv::CAP_DSHOW);
#endif
#else
	m_pCvCapture->open(m_Id);
#endif
	if (!m_pCvCapture->isOpened()) return false;
	
	// Try to set capture parameters although not always works
	m_pCvCapture->set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(m_Width));
	m_pCvCapture->set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(m_Height));
#if defined(EVIACAM_MODERN_CAPTURE)
    m_pCvCapture->set(cv::CAP_PROP_FPS, static_cast<double>(m_FrameRate));
#endif
	// The following line does nothing under MS Windows
	//m_pCvCapture->set(cv::CAP_PROP_FPS, static_cast<double>(m_FrameRate));

	return true;
}

void CCameraCV::DoClose ()
{
	m_pCvCapture->release();
}

bool CCameraCV::DoQueryFrame(cv::Mat& frame)
{
	assert (m_pCvCapture->isOpened());
	if (!m_pCvCapture->isOpened()) return false; // FIXME(silhusk): could remove this check?

#if defined(EVIACAM_MODERN_CAPTURE)
    const char* replay = std::getenv("EVIACAM_REPLAY");
    if (replay && *replay) {
        const auto now = static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        if (m_nextReplayTick > now)
            std::this_thread::sleep_for(std::chrono::milliseconds(m_nextReplayTick - now));
        m_nextReplayTick = (m_nextReplayTick > now ? m_nextReplayTick : now) + 33;
        if (m_pCvCapture->read(frame)) return true;
        m_pCvCapture->set(cv::CAP_PROP_POS_FRAMES, 0);
        return m_pCvCapture->read(frame);
    }
#endif
	if (!m_pCvCapture->read(frame)) {
		return false;
	};
	
#if defined(linux)
	// FIXME(silhusk): reimplement for `frame` instead of `pImage`
	// It seems that under Linux no proper channelSeq is reported
	// Tested with Logitech Quickcam pro 4000 
	pImage->channelSeq[0]= 'B';
	pImage->channelSeq[2]= 'R';
#endif

	return true;
}

int CCameraCV::GetNumDevices()
{
	if (!g_cvInitialized) {		
		g_cvInitialized= true; 

#if defined(EVIACAM_MODERN_CAPTURE)
        const char* replay = std::getenv("EVIACAM_REPLAY");
        if (replay && *replay) {
            strcpy(g_deviceNames[0], "Offline video (no camera)");
            return g_numDevices = 1;
        }
#ifdef __APPLE__
        for (const auto& name : eviacam::MacCameraNames()) {
            if (g_numDevices == MAX_CV_DEVICES) break;
            snprintf(g_deviceNames[g_numDevices], 50, " (Id:%d) %.38s", g_numDevices, name.c_str());
            ++g_numDevices;
        }
#else
        // Enumerate device names without opening a camera or changing its state.
        const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ICreateDevEnum* devices = nullptr;
        IEnumMoniker* enumerator = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_ICreateDevEnum, reinterpret_cast<void**>(&devices)))) {
            if (devices->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumerator, 0) == S_OK) {
                IMoniker* moniker = nullptr;
                while (g_numDevices < MAX_CV_DEVICES && enumerator->Next(1, &moniker, nullptr) == S_OK) {
                    IPropertyBag* properties = nullptr;
                    sprintf(g_deviceNames[g_numDevices], "Camera (Id:%d)", g_numDevices);
                    if (SUCCEEDED(moniker->BindToStorage(nullptr, nullptr, IID_IPropertyBag,
                                                        reinterpret_cast<void**>(&properties)))) {
                        VARIANT name;
                        VariantInit(&name);
                        if (SUCCEEDED(properties->Read(L"FriendlyName", &name, nullptr)) && name.vt == VT_BSTR) {
                            char friendlyName[256] = {};
                            WideCharToMultiByte(CP_ACP, 0, name.bstrVal, -1,
                                                friendlyName, 255, nullptr, nullptr);
                            snprintf(g_deviceNames[g_numDevices], 50, " (Id:%d) %.38s",
                                     g_numDevices, friendlyName);
                            g_deviceNames[g_numDevices][49] = 0;
                        }
                        VariantClear(&name);
                        properties->Release();
                    }
                    ++g_numDevices;
                    moniker->Release();
                }
                enumerator->Release();
            }
            devices->Release();
        }
        if (SUCCEEDED(init)) CoUninitialize();
#endif
#else

		int i;
		cv::VideoCapture tmpCapture;

		// Detect number of connected devices
		for (i= 0; i< MAX_CV_DEVICES; ++i) {
			if (!tmpCapture.open(i)) break;

			tmpCapture.release();

			// Generate device name
			sprintf (g_deviceNames[i], "Camera (Id:%d)", i);
		}		
		g_numDevices= i;
#endif
	}
	return g_numDevices;
}

#if defined(EVIACAM_MODERN_CAPTURE)
bool CCameraCV::HasSettingsDialog() {
#ifdef __APPLE__
    return false; // AVFoundation has no driver settings dialog.
#else
    const char* replay = std::getenv("EVIACAM_REPLAY");
    return !(replay && *replay);
#endif
}
void CCameraCV::ShowSettingsDialog() {
    if (HasSettingsDialog() && m_pCvCapture->isOpened())
        m_pCvCapture->set(cv::CAP_PROP_SETTINGS, 0);
}
#endif

char* CCameraCV::GetDeviceName (unsigned int id)
{
	if ((int) id>= GetNumDevices()) return NULL;	
	return g_deviceNames[id];
}
