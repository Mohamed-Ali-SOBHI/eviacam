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
#include "config.h"
#include "visionpipeline.h"
#include "eviacamapp.h"
#include "viacamcontroller.h"

#include "crvimage.h"
#include "timeutil.h"
#include "paths.h"
#include "simplelog.h"

#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#if defined(ENABLE_YUNET_FACE_DETECTOR)
#include <opencv2/objdetect.hpp>
#endif

#include <math.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <wx/filename.h>
#include <wx/msgdlg.h>
#include <wx/stdpaths.h>

// Constants
#define DEFAULT_TRACK_AREA_WIDTH_PERCENT 0.50f
#define DEFAULT_TRACK_AREA_HEIGHT_PERCENT 0.30f
#define DEFAULT_TRACK_AREA_X_CENTER_PERCENT 0.5f
#define DEFAULT_TRACK_AREA_Y_CENTER_PERCENT 0.5f
#define DEFAULT_FACE_DETECTION_TIMEOUT 5000
#define COLOR_DEGRADATION_TIME 5000

using namespace cv;

namespace {

const float YUNET_SCORE_THRESHOLD = 0.9f;
const float YUNET_NMS_THRESHOLD = 0.3f;
const int YUNET_TOP_K = 5000;
const int YUNET_MAX_INPUT_SIDE = 640;
const cv::Size HAAR_MIN_FACE_SIZE(65, 65);

struct FaceDetectionResult {
	cv::Rect box;
	float score;
};

static std::string ToUtf8(const wxString& value)
{
	return std::string(value.mb_str(wxConvUTF8));
}

static bool safeHaarCascadeLoad(cv::CascadeClassifier& c, const char* fileName)
{
	std::string fileName0(fileName);
	bool result = false;

	try {
		result = c.load(fileName0);
	}
	catch (cv::Exception& e) {
		SLOG_WARNING("Cannot load haar cascade: %s", e.what());
	}
	catch (...) {
		SLOG_WARNING("Error loading haar cascade");
	}

	return result;
}

static wxString GetExecutableDir()
{
	return wxFileName(wxStandardPaths::Get().GetExecutablePath()).GetPath();
}

static std::vector<wxString> GetPackagedFileCandidates(const wxString& fileName)
{
	std::vector<wxString> candidates;
	const wxString dataDir = eviacam::GetDataDir();
	const wxString executableDir = GetExecutableDir();
	const wxString currentDir = wxFileName::GetCwd();

	if (!dataDir.IsEmpty()) {
		candidates.push_back(dataDir + wxT("/") + fileName);
		candidates.push_back(dataDir + wxT("/data/") + fileName);
	}
	if (!executableDir.IsEmpty()) {
		candidates.push_back(executableDir + wxT("/") + fileName);
		candidates.push_back(executableDir + wxT("/data/") + fileName);
		candidates.push_back(executableDir + wxT("/../src/data/") + fileName);
	}
	if (!currentDir.IsEmpty()) {
		candidates.push_back(currentDir + wxT("/src/data/") + fileName);
	}

	return candidates;
}

static wxString FindFirstExistingFile(const std::vector<wxString>& candidates)
{
	for (size_t i = 0; i < candidates.size(); ++i) {
		if (wxFileName::FileExists(candidates[i])) return candidates[i];
	}

	return wxString();
}

static cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& imageSize)
{
	return rect & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

static cv::Point2f GetRectCenter(const cv::Rect& rect)
{
	return cv::Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
}

static bool SelectFaceDetection(
	const std::vector<FaceDetectionResult>& detections,
	const cv::Rect& currentTrackArea,
	bool preferCurrentFace,
	cv::Rect& selectedFace)
{
	if (detections.empty()) return false;

	size_t selectedIndex = 0;
	if (preferCurrentFace && currentTrackArea.area() > 0) {
		const cv::Point2f currentCenter = GetRectCenter(currentTrackArea);
		float bestDistance = std::numeric_limits<float>::max();

		for (size_t i = 0; i < detections.size(); ++i) {
			const cv::Point2f detectionCenter = GetRectCenter(detections[i].box);
			const float dx = detectionCenter.x - currentCenter.x;
			const float dy = detectionCenter.y - currentCenter.y;
			const float distance = dx * dx + dy * dy;

			if (distance < bestDistance) {
				bestDistance = distance;
				selectedIndex = i;
			}
		}
	}
	else {
		float bestScore = detections[0].score;
		int bestArea = detections[0].box.area();

		for (size_t i = 1; i < detections.size(); ++i) {
			const int candidateArea = detections[i].box.area();
			if (detections[i].score > bestScore ||
				(detections[i].score == bestScore && candidateArea > bestArea)) {
				bestScore = detections[i].score;
				bestArea = candidateArea;
				selectedIndex = i;
			}
		}
	}

	selectedFace = detections[selectedIndex].box;
	return true;
}

class FaceDetectionBackend {
public:
	virtual ~FaceDetectionBackend() {}
	virtual const char* GetName() const = 0;
	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		cv::Rect& selectedFace) = 0;
};

class HaarFaceDetector : public FaceDetectionBackend {
public:
	static std::unique_ptr<FaceDetectionBackend> Create(wxString& modelPath, wxString& error)
	{
		std::vector<wxString> candidates = GetPackagedFileCandidates(
			wxT("haarcascade_frontalface_default.xml"));
		candidates.push_back(wxT("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"));
		candidates.push_back(wxT("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"));
		candidates.push_back(wxT("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"));

		std::unique_ptr<HaarFaceDetector> detector(new HaarFaceDetector());
		for (size_t i = 0; i < candidates.size(); ++i) {
			if (!wxFileName::FileExists(candidates[i])) continue;
			if (safeHaarCascadeLoad(detector->m_cascade, candidates[i].mb_str(wxConvUTF8))) {
				modelPath = candidates[i];
				error.clear();
				return std::unique_ptr<FaceDetectionBackend>(detector.release());
			}
		}

		error = wxT("Could not load haarcascade_frontalface_default.xml");
		return std::unique_ptr<FaceDetectionBackend>();
	}

	virtual const char* GetName() const { return "Haar"; }

	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		cv::Rect& selectedFace)
	{
		cv::Mat gray;
		if (image.channels() == 1) gray = image;
		else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		std::vector<cv::Rect> faces;
		m_cascade.detectMultiScale(
			gray,
			faces,
			1.5,
			2,
			cv::CASCADE_DO_CANNY_PRUNING,
			HAAR_MIN_FACE_SIZE);

		std::vector<FaceDetectionResult> results;
		for (size_t i = 0; i < faces.size(); ++i) {
			const cv::Rect face = ClampRect(faces[i], image.size());
			if (face.area() <= 0) continue;

			FaceDetectionResult result;
			result.box = face;
			result.score = 0.0f;
			results.push_back(result);
		}

		return SelectFaceDetection(results, currentTrackArea, preferCurrentFace, selectedFace);
	}

private:
	HaarFaceDetector() {}

	cv::CascadeClassifier m_cascade;
};

#if defined(ENABLE_YUNET_FACE_DETECTOR)
class YuNetFaceDetector : public FaceDetectionBackend {
public:
	static std::unique_ptr<FaceDetectionBackend> Create(wxString& modelPath, wxString& error)
	{
		const wxString yuNetPath = FindFirstExistingFile(
			GetPackagedFileCandidates(wxT("face_detection_yunet_2023mar.onnx")));
		if (yuNetPath.IsEmpty()) {
			error = wxT("Could not find face_detection_yunet_2023mar.onnx");
			return std::unique_ptr<FaceDetectionBackend>();
		}

		try {
			std::unique_ptr<YuNetFaceDetector> detector(new YuNetFaceDetector());
			detector->m_detector = cv::FaceDetectorYN::create(
				ToUtf8(yuNetPath),
				"",
				cv::Size(320, 320),
				YUNET_SCORE_THRESHOLD,
				YUNET_NMS_THRESHOLD,
				YUNET_TOP_K);

			modelPath = yuNetPath;
			error.clear();
			return std::unique_ptr<FaceDetectionBackend>(detector.release());
		}
		catch (const cv::Exception& e) {
			error = wxString(e.what(), wxConvUTF8);
		}
		catch (...) {
			error = wxT("Failed to initialize YuNet backend");
		}

		return std::unique_ptr<FaceDetectionBackend>();
	}

	virtual const char* GetName() const { return "YuNet"; }

	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		cv::Rect& selectedFace)
	{
		if (image.empty()) return false;

		double scale = 1.0;
		cv::Mat resized = image;
		const int maxSide = std::max(image.cols, image.rows);
		if (maxSide > YUNET_MAX_INPUT_SIDE) {
			scale = static_cast<double>(YUNET_MAX_INPUT_SIDE) / static_cast<double>(maxSide);
			cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
		}

		m_detector->setInputSize(resized.size());

		cv::Mat detections;
		m_detector->detect(resized, detections);
		if (detections.empty()) return false;

		std::vector<FaceDetectionResult> results;
		for (int row = 0; row < detections.rows; ++row) {
			const float score = detections.at<float>(row, 14);
			if (score < YUNET_SCORE_THRESHOLD) continue;

			const int x = cvRound(detections.at<float>(row, 0) / scale);
			const int y = cvRound(detections.at<float>(row, 1) / scale);
			const int width = cvRound(detections.at<float>(row, 2) / scale);
			const int height = cvRound(detections.at<float>(row, 3) / scale);
			const cv::Rect face = ClampRect(cv::Rect(x, y, width, height), image.size());
			if (face.area() <= 0) continue;

			FaceDetectionResult result;
			result.box = face;
			result.score = score;
			results.push_back(result);
		}

		return SelectFaceDetection(results, currentTrackArea, preferCurrentFace, selectedFace);
	}

private:
	YuNetFaceDetector() {}

	cv::Ptr<cv::FaceDetectorYN> m_detector;
};
#endif

static std::unique_ptr<FaceDetectionBackend> CreateFaceDetectionBackend(bool& available)
{
	available = false;

#if defined(ENABLE_YUNET_FACE_DETECTOR)
	wxString yuNetPath;
	wxString yuNetError;
	std::unique_ptr<FaceDetectionBackend> backend = YuNetFaceDetector::Create(yuNetPath, yuNetError);
	if (backend.get() != NULL) {
		SLOG_INFO("Using face detector backend: %s (%s)", backend->GetName(), ToUtf8(yuNetPath).c_str());
		available = true;
		return backend;
	}

	SLOG_WARNING(
		"YuNet face detector unavailable, falling back to Haar: %s",
		ToUtf8(yuNetError).c_str());
#else
	SLOG_INFO("YuNet face detector support not enabled in this build; using Haar fallback");
#endif

	wxString haarPath;
	wxString haarError;
	std::unique_ptr<FaceDetectionBackend> backend = HaarFaceDetector::Create(haarPath, haarError);
	if (backend.get() != NULL) {
		SLOG_INFO("Using face detector backend: %s (%s)", backend->GetName(), ToUtf8(haarPath).c_str());
		available = true;
		return backend;
	}

	SLOG_WARNING("Haar face detector unavailable: %s", ToUtf8(haarError).c_str());
	return std::unique_ptr<FaceDetectionBackend>();
}

} // namespace

CVisionPipeline::CVisionPipeline(wxThreadKind kind)
: wxThread(kind)
// Actually it is not needed all the features a condition object offers, but
// we use it because we need a timeout based wait call. The associated mutex
// is not used at all.
, m_condition(m_mutex)
, m_faceDetectionAvailable(false)
, m_faceLocationStatus(0) // 0 -> not available, 1 -> available
{
	InitDefaults();

	m_isRunning = false;
	m_trackAreaTimeout.SetWaitTimeMs(COLOR_DEGRADATION_TIME);

	m_faceDetector = CreateFaceDetectionBackend(m_faceDetectionAvailable);
	if (!m_faceDetectionAvailable) {
		wxMessageDialog dlg(NULL, _("The face localization option is not enabled."),
			_T("Enable Viacam"), wxICON_ERROR | wxOK);
		dlg.ShowModal();
		return;
	}

	// Create and start face detection thread
	if (Create() == wxTHREAD_NO_ERROR) {
#if defined(WIN32)
		// On linux this ends up calling setpriority syscall which changes
		// the priority of the whole process :-( (see wxWidgets threadpsx.cpp)
		// TODO: implement it using pthreads
		SetPriority(WXTHREAD_MIN_PRIORITY);
#endif
		m_isRunning = true;
		Run();
	}
}

CVisionPipeline::~CVisionPipeline()
{
	if (m_isRunning) {
		m_isRunning = false;
		m_condition.Signal();
		Wait();
	}
}

wxThreadError CVisionPipeline::Create(unsigned int stackSize)
{
	return wxThread::Create(stackSize);
}

// Low-priority secondary thread where face localization occurs
wxThread::ExitCode CVisionPipeline::Entry()
{
	unsigned long ts1 = 0;
	for (;;) {
		m_condition.WaitTimeout(1000);
		if (!m_isRunning) {
			break;
		}

		unsigned long now = CTimeUtil::GetMiliCount();
		if (now - ts1 >= (unsigned long) m_threadPeriod) {
			ts1 = CTimeUtil::GetMiliCount();
			if (m_imgPrevColor.empty()) continue;
			m_imageCopyMutex.Enter();
			m_imgPrevColor.copyTo(m_imgThread);
			m_imageCopyMutex.Leave();

			ComputeFaceTrackArea(m_imgThread);
		}
	}
	return 0;
}

void CVisionPipeline::ComputeFaceTrackArea(const cv::Mat& image)
{
	if (!m_trackFace) return;
	if (!m_faceDetectionAvailable || m_faceDetector.get() == NULL) return;
	if (m_faceLocationStatus) return;	// Already available

	cv::Rect currentTrackArea;
	m_trackArea.GetBoxImg(image, currentTrackArea);

	cv::Rect detectedFace;
	if (m_faceDetector->Detect(image, currentTrackArea, IsFaceDetected(), detectedFace)) {
		m_faceLocation = detectedFace;
		m_faceLocationStatus = 1;

		m_waitTime.Reset();
		m_trackAreaTimeout.Reset();
	}
}

bool CVisionPipeline::IsFaceDetected() const
{
	return !m_waitTime.HasExpired();
}

static void DrawCorners(
	cv::Mat& image,
	const std::vector<Point2f> corners,
	const cv::Scalar color)
{
	for (int i = 0; i < corners.size(); i++)
		cv::circle(image, corners[i], 1, color);
}

void CVisionPipeline::NewTracker(cv::Mat &image, float &xVel, float &yVel)
{
	cv::Rect2f trackArea;
	bool updateFeatures = false;

	// Face location has been updated?
	if (m_faceLocationStatus) {
		trackArea = m_faceLocation;
		m_faceLocationStatus = 0;
		updateFeatures = true;
	}
	else {
		cv::Rect box;
		m_trackArea.GetBoxImg(image, box);
		trackArea = box;
				
        // Need to update corners?
		if (m_corners.size()< NUM_CORNERS) updateFeatures = true;
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
        if (m_corners.size()) {
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

bool CVisionPipeline::ProcessImage(cv::Mat& image, float& xVel, float& yVel)
{
	try {
		cv::Mat detectFrame = image.clone();
		cv::cvtColor(image, m_imgCurr, cv::COLOR_BGR2GRAY);
		
		// Initialize on first frame
		if (m_imgPrev.empty()) {
			m_imgCurr.copyTo(m_imgPrev);
			m_imgPrevColor = detectFrame;
		}
		
		// TODO: fine grained synchronization
		m_imageCopyMutex.Enter();

		NewTracker(image, xVel, yVel);

		// Store current image as previous
		cv::swap(m_imgPrev, m_imgCurr);
		m_imgPrevColor = detectFrame;
		m_imageCopyMutex.Leave();

		// Notifies face detection thread when needed
		if (m_trackFace && m_faceDetectionAvailable) {
			m_trackArea.SetDegradation(255 - m_trackAreaTimeout.PercentagePassed() * 255 / 100);
			m_condition.Signal();
		}

		if (m_trackFace &&
			m_faceDetectionAvailable &&
			m_enableWhenFaceDetected &&
			!IsFaceDetected())
			return false;
		else
			return true;
	}
	catch (const std::exception& e) {
		SLOG_ERR("Exception: %s\n", e.what());
		exit(1);
	}

	return false;
}

enum ECpuValues { LOWEST = 1500, LOW = 800, NORMAL = 400, HIGH = 100, HIGHEST = 0 };

int CVisionPipeline::GetCpuUsage ()
{
	switch (m_threadPeriod)
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
			m_threadPeriod= LOWEST;
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

	pConfObj->Write(_T("trackFace"), m_trackFace);
	pConfObj->Write(_T("enableWhenFaceDetected"), m_enableWhenFaceDetected);
	pConfObj->Write(_T("locateFaceTimeout"), (int) m_waitTime.GetWaitTimeMs());
	pConfObj->Write(_T("threadPeriod"), (int) m_threadPeriod);

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
	pConfObj->Read(_T("trackFace"), &m_trackFace);
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
