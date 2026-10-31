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
#include "visionpipeline.h"

#include "crvimage.h"
#include "timeutil.h"
#include "paths.h"
#include "simplelog.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/video/tracking.hpp>

#include <math.h>
#include <wx/msgdlg.h>
#include <wx/stdpaths.h>
#include <wx/utils.h>

// Constants
#define DEFAULT_TRACK_AREA_WIDTH_PERCENT 0.50f
#define DEFAULT_TRACK_AREA_HEIGHT_PERCENT 0.30f
#define DEFAULT_TRACK_AREA_X_CENTER_PERCENT 0.5f
#define DEFAULT_TRACK_AREA_Y_CENTER_PERCENT 0.5f
#define DEFAULT_FACE_DETECTION_TIMEOUT 5000
#define COLOR_DEGRADATION_TIME 5000

using namespace cv;

CVisionPipeline::CVisionPipeline (wxThreadKind kind) 
: wxThread (kind)
// Frame-ready notifications and shutdown are protected by the condition mutex.
, m_condition(m_mutex)
, m_faceLocationStatus(0) // 0 -> not available, 1 -> available
{
	InitDefaults();

	m_isRunning= false;
	m_trackAreaTimeout.SetWaitTimeMs(COLOR_DEGRADATION_TIME);

	//
	// Load YuNet, with the original Haar detector as an explicit mode/fallback.
	// 
    const wxString dataDir = eviacam::GetDataDir();
    wxString detectorMode;
    wxGetEnv(wxT("EVIACAM_FACE_DETECTOR"), &detectorMode);
    m_detectorReady = m_faceDetector.Load(
        std::string((dataDir + wxT("/haarcascade_frontalface_default.xml")).utf8_str()),
        std::string((dataDir + wxT("/face_detection_yunet_2023mar.onnx")).utf8_str()),
        detectorMode == wxT("haar"));
    if (!m_faceDetector.LastError().empty())
        SLOG_WARNING("Face detector: %s", m_faceDetector.LastError().c_str());
    SLOG_INFO("Face detector: %s", m_faceDetector.UsingYuNet() ? "YuNet" : "Haar");
    if (!m_detectorReady) return;
    m_yunetActive = m_faceDetector.UsingYuNet();

	// Create and start face detection thread
	if (Create() == wxTHREAD_NO_ERROR) {
#if defined (WIN32)
		// On linux this ends up calling setpriority syscall which changes
		// the priority of the whole process :-( (see wxWidgets threadpsx.cpp)
		// TODO: implement it using pthreads
		SetPriority (WXTHREAD_MIN_PRIORITY);
#endif
		m_isRunning= true;
		m_threadStarted = (Run() == wxTHREAD_NO_ERROR);
		if (!m_threadStarted) m_isRunning = false;
	}
}

CVisionPipeline::~CVisionPipeline () {
    if (m_threadStarted) {
        {
            wxMutexLocker lock(m_mutex);
            m_isRunning = false;
            m_condition.Signal();
        }
        Wait();
    }
}

wxThreadError CVisionPipeline::Create(unsigned int stackSize) {
    return wxThread::Create(stackSize);
}

wxThread::ExitCode CVisionPipeline::Entry() {
    unsigned long lastDetection = 0;
    while (m_isRunning) {
        {
            wxMutexLocker lock(m_mutex);
            while (m_isRunning && !m_frameReady) m_condition.WaitTimeout(1000);
            if (!m_isRunning) break;
            m_frameReady = false;
        }
        const unsigned long now = CTimeUtil::GetMiliCount();
        if (now - lastDetection < static_cast<unsigned long>(m_threadPeriod.load())) continue;
        lastDetection = now;
        {
            wxCriticalSectionLocker lock(m_imageCopyMutex);
            m_detectionFrame.copyTo(m_imgThread);
        }
        if (!m_imgThread.empty()) ComputeFaceTrackArea(m_imgThread);
    }
    return 0;
}

void CVisionPipeline::ComputeFaceTrackArea(const cv::Mat& image) {
    if (!m_trackFace) return;
    const unsigned long started = CTimeUtil::GetMiliCount();
    cv::Rect previous;
    {
        wxCriticalSectionLocker lock(m_faceMutex);
        if (m_faceLocationStatus) return;
        if (started - m_faceTimestamp < 1500) previous = m_faceLocation;
    }
    try {
        const bool wasYuNet = m_faceDetector.UsingYuNet();
        const auto faces = m_faceDetector.Detect(image);
        m_yunetActive = m_faceDetector.UsingYuNet();
        if (wasYuNet && !m_faceDetector.UsingYuNet())
            SLOG_WARNING("YuNet failed; using Haar: %s", m_faceDetector.LastError().c_str());
        const cv::Rect selected = wasYuNet ? eviacam::FaceDetector::Select(faces, previous)
                                          : (faces.empty() ? cv::Rect() : faces.front().box);
        if (selected.area() > 0) {
            wxCriticalSectionLocker lock(m_faceMutex);
            m_faceLocation = selected;
            m_faceTimestamp = started;
            m_faceLocationStatus = 1;
        }
    } catch (const cv::Exception& e) { SLOG_WARNING("Face detection: %s", e.what()); }
}

bool CVisionPipeline::IsFaceDetected () const
{
	return m_hasDetectedFace && !m_waitTime.HasExpired();
}

static 
void DrawCorners(cv::Mat& image, const std::vector<Point2f> corners, const cv::Scalar color) {
	for (int i = 0; i < corners.size(); i++)
		cv::circle(image, corners[i], 1, color);
}

void CVisionPipeline::NewTracker(cv::Mat &image, float &xVel, float &yVel)
{
	cv::Rect2f trackArea;
	bool updateFeatures = false;

    cv::Rect box;
    m_trackArea.GetBoxImg(image, box);
    trackArea = box;
    updateFeatures = m_corners.size() < NUM_CORNERS;
    {
        wxCriticalSectionLocker lock(m_faceMutex);
        if (m_faceLocationStatus) {
            if (CTimeUtil::GetMiliCount() - m_faceTimestamp <= 500 && m_trackFace) {
                trackArea = m_faceLocation & cv::Rect(0, 0, image.cols, image.rows);
                updateFeatures = true;
                m_waitTime.Reset();
                m_trackAreaTimeout.Reset();
                m_hasDetectedFace = true;
            }
            m_faceLocationStatus = 0;
        }
    }
    trackArea = cv::Rect(trackArea) & cv::Rect(0, 0, image.cols, image.rows);
    if (trackArea.width < 12 || trackArea.height < 12) {
        xVel = yVel = 0;
        m_corners.clear();
        return;
    }

	if (updateFeatures) {
		// 
		// Set smaller area to extract features to track
		//
		#define SMALL_AREA_RATIO 0.4f

		cv::Rect2f featuresTrackArea;
		featuresTrackArea.x = trackArea.x + 
			trackArea.width * ((1.0f - SMALL_AREA_RATIO) / 2.0f);
		featuresTrackArea.y = trackArea.y + 
			trackArea.height * ((1.0f - SMALL_AREA_RATIO) / 2.0f);
		featuresTrackArea.width = trackArea.width * SMALL_AREA_RATIO;
		featuresTrackArea.height = trackArea.height * SMALL_AREA_RATIO;

		//
		// Find features to track
		//
		#define QUALITY_LEVEL  0.001   // 0.01
		#define MIN_DISTANTE 2

        cv::Mat prevImg = m_imgPrev(featuresTrackArea);
		
        goodFeaturesToTrack(prevImg, m_corners, NUM_CORNERS, QUALITY_LEVEL, MIN_DISTANTE);
        TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
        if (m_corners.size() && prevImg.cols >= 15 && prevImg.rows >= 15) {
		    cornerSubPix(prevImg, m_corners, Size(5, 5), Size(-1, -1), termcrit);
        }
		
		//
		// Update features location
		//
		for (int i = 0; i < m_corners.size(); i++) {
			m_corners[i].x += featuresTrackArea.x;
			m_corners[i].y += featuresTrackArea.y;
		}
	}

	if (slog_get_priority() >= SLOG_PRIO_DEBUG) {
	    DrawCorners(image, m_corners, cv::Scalar(255, 0, 0));
    }

	//
	// Track corners
	//
    cv::Mat prevImg = m_imgPrev(trackArea);
    cv::Mat currImg = m_imgCurr(trackArea);

	// Update corners location for the new ROI
	for (int i = 0; i < m_corners.size(); i++) {
		m_corners[i].x -= trackArea.x;
		m_corners[i].y -= trackArea.y;
	}

	vector<Point2f> new_corners;
	vector<uchar> status;
    vector<float> err;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,14,0.03);
    if (m_corners.size()) {
        calcOpticalFlowPyrLK(
            prevImg, currImg, m_corners, new_corners, status, err, Size(11, 11), 0, termcrit);
    }

	//
	// Accumulate motion (TODO: remove outliers?)
	//	
	int valid_corners = 0;
	float dx = 0, dy = 0;

	for (int i = 0; i< m_corners.size(); i++) {
		if (status[i] &&
			m_corners[i].x >= 0 &&
			m_corners[i].x < trackArea.width &&
			m_corners[i].y >= 0 &&
			m_corners[i].y < trackArea.height) {
			dx += m_corners[i].x - new_corners[i].x;
			dy += m_corners[i].y - new_corners[i].y;

			// Update corner location, relative to full `image`
			new_corners[i].x += trackArea.x;
			new_corners[i].y += trackArea.y;

			// Save new corner location
			m_corners[valid_corners++] = new_corners[i];
		}
	}
    m_corners.resize(valid_corners);

	if (valid_corners) {
		dx = dx / (float) valid_corners;
		dy = dy / (float) valid_corners;

		xVel = 2.0 * dx;
		yVel = 2.0 * -dy;
	}
	else {
		xVel = yVel = 0;
	}

	//
	// Update tracking area location
	//
	if (m_trackFace) {
		trackArea.x -= dx;
		trackArea.y -= dy;
	}
	
	//
	// Update visible tracking area
	//
	m_trackArea.SetSizeImg(image, cvRound(trackArea.width), cvRound(trackArea.height));
	m_trackArea.SetCenterImg(image, 
		cvRound(trackArea.x + trackArea.width / 2.0f),
		cvRound(trackArea.y + trackArea.height / 2.0f));

	//
	// Draw corners
	//
	DrawCorners(image, m_corners, cv::Scalar(0, 255, 0));
}

bool CVisionPipeline::ProcessImage (cv::Mat& image, float& xVel, float& yVel)
{
	xVel = yVel = 0;
	if (image.empty()) return false;
	try {
		cv::cvtColor(image, m_imgCurr, cv::COLOR_BGR2GRAY);
		
		// Initialize on first frame
		if (m_imgPrev.empty() || m_imgPrev.size() != m_imgCurr.size()) {
			m_imgCurr.copyTo(m_imgPrev);
			m_corners.clear();
		}
		
		// TODO: fine grained synchronization
		{
        wxCriticalSectionLocker lock(m_imageCopyMutex);
        image.copyTo(m_detectionFrame);

		NewTracker(image, xVel, yVel);

		// Store current image as previous
		cv::swap(m_imgPrev, m_imgCurr);
		}

		// Notifies face detection thread when needed
		if (m_trackFace) {
			m_trackArea.SetDegradation(255 - m_trackAreaTimeout.PercentagePassed() * 255 / 100);
			wxMutexLocker lock(m_mutex);
            m_frameReady = true;
            m_condition.Signal();
		}

		if (m_trackFace && m_enableWhenFaceDetected && !IsFaceDetected())
			return false;
		else
			return true;
	}
	catch (const std::exception& e) {
		SLOG_ERR("Exception: %s\n", e.what());
		xVel = yVel = 0;
		m_corners.clear();
		m_imgPrev.release();
		return false;
	}

	return false;
}

enum ECpuValues { LOWEST = 1500, LOW = 800, NORMAL = 400, HIGH = 100, HIGHEST = 0 };

int CVisionPipeline::GetCpuUsage ()
{
	switch (m_threadPeriod.load())
	{
		case LOWEST:
			return (int) CVisionPipeline::ECpuUsage(CPU_LOWEST);
			break;
		case LOW:
			return (int) CVisionPipeline::ECpuUsage(CPU_LOW);
			break;
		case HIGH:
			return (int) CVisionPipeline::ECpuUsage(CPU_HIGH);
			break;
		case HIGHEST:
			return (int) CVisionPipeline::ECpuUsage(CPU_HIGHEST);
			break;
		default:
			return (int) CVisionPipeline::ECpuUsage(CPU_NORMAL);
			break;
	}
}

void CVisionPipeline::SetCpuUsage (int value)
{
	switch (value)
	{
		case (int) CVisionPipeline::ECpuUsage(CPU_LOWEST):
			SetThreadPeriod(LOWEST);
			break;
		case (int) CVisionPipeline::ECpuUsage(CPU_LOW):
			SetThreadPeriod(LOW);
			break;
		case (int) CVisionPipeline::ECpuUsage(CPU_NORMAL):
			SetThreadPeriod(NORMAL);
			break;
		case (int) CVisionPipeline::ECpuUsage(CPU_HIGH):
			SetThreadPeriod(HIGH);
			break;
		case (int) CVisionPipeline::ECpuUsage(CPU_HIGHEST):
			SetThreadPeriod(HIGHEST);
			break;
	}
}

void CVisionPipeline::SetThreadPeriod (int value)
{
	switch (value)
	{
		case LOWEST:
			m_threadPeriod= LOWEST;
			break;
		case LOW:
			m_threadPeriod= LOW;
			break;
		case HIGH:
			m_threadPeriod= HIGH;
			break;
		case HIGHEST:
			m_threadPeriod= HIGHEST;
			break;
		default:
			m_threadPeriod= NORMAL;
			break;
	}
}

//
// Configuration methods
//
void CVisionPipeline::InitDefaults()
{
	m_trackFace= true;
	m_enableWhenFaceDetected= false;
	m_waitTime.SetWaitTimeMs(DEFAULT_FACE_DETECTION_TIMEOUT);
	SetThreadPeriod(CPU_NORMAL);
	m_trackArea.SetSize (DEFAULT_TRACK_AREA_WIDTH_PERCENT, DEFAULT_TRACK_AREA_HEIGHT_PERCENT);
	m_trackArea.SetCenter (DEFAULT_TRACK_AREA_X_CENTER_PERCENT, DEFAULT_TRACK_AREA_Y_CENTER_PERCENT);
	
}

void CVisionPipeline::WriteProfileData(wxConfigBase* pConfObj)
{
	float xc, yc, width, height;

	pConfObj->SetPath (_T("motionTracker"));	

	pConfObj->Write(_T("trackFace"), m_trackFace.load());
	pConfObj->Write(_T("enableWhenFaceDetected"), m_enableWhenFaceDetected);
	pConfObj->Write(_T("locateFaceTimeout"), (int) m_waitTime.GetWaitTimeMs());
	pConfObj->Write(_T("threadPeriod"), m_threadPeriod.load());

	m_trackArea.GetSize (width, height);
	
	pConfObj->Write (_T("trackAreaWidth"), (double) width);
	pConfObj->Write (_T("trackAreaHeight"), (double) height);

	if (!m_trackFace) {		
		m_trackArea.GetCenter (xc, yc);
		pConfObj->Write (_T("trackAreaCenterX"), (double) xc);
		pConfObj->Write (_T("trackAreaCenterY"), (double) yc);
	}

	pConfObj->SetPath (_T(".."));
}

void CVisionPipeline::ReadProfileData(wxConfigBase* pConfObj)
{
	// Ensure proper default values if read fails
	double	xc= DEFAULT_TRACK_AREA_X_CENTER_PERCENT, 
		yc= DEFAULT_TRACK_AREA_Y_CENTER_PERCENT,
                width= DEFAULT_TRACK_AREA_WIDTH_PERCENT,
		height= DEFAULT_TRACK_AREA_HEIGHT_PERCENT;
				
	int locateFaceTimeout = DEFAULT_FACE_DETECTION_TIMEOUT;
	int threadPeriod = -1;

	pConfObj->SetPath (_T("motionTracker"));
	bool trackFace = m_trackFace.load();
	pConfObj->Read(_T("trackFace"), &trackFace);
	m_trackFace = trackFace;
	pConfObj->Read(_T("enableWhenFaceDetected"), &m_enableWhenFaceDetected);
	pConfObj->Read(_T("locateFaceTimeout"), &locateFaceTimeout);
	pConfObj->Read (_T("trackAreaWidth"), &width);
	pConfObj->Read (_T("trackAreaHeight"), &height);
	pConfObj->Read (_T("threadPeriod"), &threadPeriod);
	
	SetThreadPeriod(threadPeriod);
	
	m_trackArea.SetSize ((float) width, (float)height);
	m_waitTime.SetWaitTimeMs(locateFaceTimeout);

	if (!m_trackFace) {
		pConfObj->Read (_T("trackAreaCenterX"), &xc);
		pConfObj->Read (_T("trackAreaCenterY"), &yc);

		m_trackArea.SetCenter ((float)xc, (float)yc);		
	}

	pConfObj->SetPath (_T(".."));
}
