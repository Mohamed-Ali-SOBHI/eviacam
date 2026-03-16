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
#include <opencv2/imgcodecs.hpp>
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
#include <cstdint>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <wx/filename.h>
#include <wx/msgdlg.h>
#include <wx/process.h>
#include <wx/socket.h>
#include <wx/stopwatch.h>
#include <wx/stdpaths.h>
#include <wx/stream.h>
#include <wx/txtstrm.h>
#include <wx/utils.h>

// Constants
#define DEFAULT_TRACK_AREA_WIDTH_PERCENT 0.50f
#define DEFAULT_TRACK_AREA_HEIGHT_PERCENT 0.30f
#define DEFAULT_TRACK_AREA_X_CENTER_PERCENT 0.5f
#define DEFAULT_TRACK_AREA_Y_CENTER_PERCENT 0.5f
#define DEFAULT_FACE_DETECTION_TIMEOUT 5000
#define COLOR_DEGRADATION_TIME 5000

using namespace cv;

enum { MIN_ACTIVE_TRACK_CORNERS = 6 };

struct FaceDetectionResult {
	cv::Rect box;
	float score;
	std::vector<cv::Point2f> landmarks;
};

class FaceDetectionBackend {
public:
	virtual ~FaceDetectionBackend() {}
	virtual const char* GetName() const = 0;
	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		FaceDetectionResult& selectedFace) = 0;
};

namespace {

const float YUNET_SCORE_THRESHOLD = 0.9f;
const float YUNET_NMS_THRESHOLD = 0.3f;
const int YUNET_TOP_K = 5000;
const int YUNET_MAX_INPUT_SIDE = 640;
const long MEDIAPIPE_READY_TIMEOUT_MS = 5000;
const long MEDIAPIPE_SOCKET_CONNECT_TIMEOUT_MS = 5000;
const long MEDIAPIPE_SOCKET_IO_TIMEOUT_MS = 5000;
const uint32_t MEDIAPIPE_MAX_RESPONSE_BYTES = 4096;
const cv::Size HAAR_MIN_FACE_SIZE(65, 65);

static std::string ToUtf8(const wxString& value)
{
	return std::string(value.mb_str(wxConvUTF8));
}

static wxUint32 TimeoutMsToSeconds(long timeoutMs)
{
	return timeoutMs <= 0
		? 0
		: static_cast<wxUint32>((timeoutMs + 999) / 1000);
}

static bool SocketWriteAll(wxSocketBase& socket, const void* data, size_t size)
{
	socket.Write(data, size);
	return !socket.Error() && static_cast<size_t>(socket.LastCount()) == size;
}

static bool SocketReadAll(wxSocketBase& socket, void* data, size_t size)
{
	socket.Read(data, size);
	return !socket.Error() && static_cast<size_t>(socket.LastCount()) == size;
}

static bool SocketSendMessage(wxSocketBase& socket, const void* data, size_t size)
{
	const uint32_t payloadSize = static_cast<uint32_t>(size);
	if (!SocketWriteAll(socket, &payloadSize, sizeof(payloadSize))) return false;
	return payloadSize == 0 || SocketWriteAll(socket, data, payloadSize);
}

static bool SocketReceiveMessage(wxSocketBase& socket, wxString& message)
{
	uint32_t payloadSize = 0;
	if (!SocketReadAll(socket, &payloadSize, sizeof(payloadSize))) return false;
	if (payloadSize == 0 || payloadSize > MEDIAPIPE_MAX_RESPONSE_BYTES) return false;

	std::vector<char> payload(payloadSize);
	if (!SocketReadAll(socket, &payload[0], payloadSize)) return false;

	payload.push_back('\0');
	message = wxString::FromUTF8(&payload[0]);
	return !message.IsEmpty();
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

static cv::Rect ClampRect(const cv::Rect2f& rect, const cv::Size& imageSize)
{
	return ClampRect(
		cv::Rect(
			cvFloor(rect.x),
			cvFloor(rect.y),
			cvCeil(rect.width),
			cvCeil(rect.height)),
		imageSize);
}

static cv::Point2f GetRectCenter(const cv::Rect& rect)
{
	return cv::Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
}

static cv::Rect BuildFeatureTrackArea(const cv::Rect2f& trackArea, const cv::Size& imageSize)
{
	// Favor upper-face features (eyes/eyebrows/nose bridge) instead of the mouth.
	cv::Rect featureArea = ClampRect(
		cv::Rect2f(
			trackArea.x + trackArea.width * 0.20f,
			trackArea.y + trackArea.height * 0.10f,
			trackArea.width * 0.60f,
			trackArea.height * 0.30f),
		imageSize);

	if (featureArea.width >= 10 && featureArea.height >= 10) {
		return featureArea;
	}

	return ClampRect(
		cv::Rect2f(
			trackArea.x + trackArea.width * 0.14f,
			trackArea.y + trackArea.height * 0.08f,
			trackArea.width * 0.72f,
			trackArea.height * 0.44f),
		imageSize);
}

static void RefineCornersInRoi(cv::Mat& image, std::vector<Point2f>& corners)
{
	if (corners.empty()) return;

	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	cornerSubPix(image, corners, Size(5, 5), Size(-1, -1), termcrit);
}

static void OffsetCorners(std::vector<Point2f>& corners, int x, int y)
{
	for (int i = 0; i < corners.size(); i++) {
		corners[i].x += x;
		corners[i].y += y;
	}
}

static void FilterLowerFaceCorners(std::vector<Point2f>& corners, const cv::Rect& trackArea)
{
	if (corners.empty()) return;

	const float maxCornerY = trackArea.y + trackArea.height * 0.62f;
	corners.erase(
		std::remove_if(
			corners.begin(),
			corners.end(),
			[maxCornerY](const Point2f& point) {
				return point.y > maxCornerY;
			}),
		corners.end());
}

static void CollectLandmarkCorners(
	const cv::Mat& image,
	const std::vector<Point2f>& landmarks,
	const cv::Rect& trackArea,
	std::vector<Point2f>& corners)
{
	corners.clear();
	if (image.empty() || landmarks.empty() || trackArea.area() <= 0) return;

	const size_t landmarkCount = std::min<size_t>(3, landmarks.size());
	const float maxCornerY = trackArea.y + trackArea.height * 0.58f;

	for (size_t i = 0; i < landmarkCount; ++i) {
		const Point2f& landmark = landmarks[i];
		if (landmark.y > maxCornerY) continue;

		cv::Rect landmarkWindow = ClampRect(
			cv::Rect(
				cvRound(landmark.x) - 18,
				cvRound(landmark.y) - 18,
				36,
				36),
			image.size());
		if (landmarkWindow.width < 8 || landmarkWindow.height < 8) continue;

		std::vector<Point2f> localCorners;
		cv::Mat landmarkImage = image(landmarkWindow);
		goodFeaturesToTrack(landmarkImage, localCorners, 6, 0.001, 2);

		if (!localCorners.empty()) {
			RefineCornersInRoi(landmarkImage, localCorners);
			OffsetCorners(localCorners, landmarkWindow.x, landmarkWindow.y);
			corners.insert(corners.end(), localCorners.begin(), localCorners.end());
		}
		else if (trackArea.contains(cv::Point(cvRound(landmark.x), cvRound(landmark.y)))) {
			corners.push_back(landmark);
		}
	}

	FilterLowerFaceCorners(corners, trackArea);
}

static bool GetLandmarkAnchor(const std::vector<Point2f>& landmarks, Point2f& anchor)
{
	if (landmarks.size() < 3) return false;

	const Point2f eyeCenter = (landmarks[0] + landmarks[1]) * 0.5f;
	anchor = Point2f(
		(eyeCenter.x * 2.0f + landmarks[2].x) / 3.0f,
		(eyeCenter.y * 2.0f + landmarks[2].y) / 3.0f);
	return true;
}

static cv::Rect BuildTrackAreaFromLandmarks(
	const std::vector<Point2f>& landmarks,
	const cv::Rect& fallbackFace,
	const cv::Size& imageSize)
{
	if (landmarks.size() < 3) {
		return ClampRect(fallbackFace, imageSize);
	}

	const Point2f rightEye = landmarks[0];
	const Point2f leftEye = landmarks[1];
	const Point2f nose = landmarks[2];
	const Point2f eyeCenter = (rightEye + leftEye) * 0.5f;
	const float eyeDistance = std::max(20.0f, static_cast<float>(norm(rightEye - leftEye)));
	const float noseDistance = std::max(12.0f, static_cast<float>(norm(nose - eyeCenter)));

	float width = eyeDistance * 2.8f;
	float height = std::max(width * 1.18f, noseDistance * 5.0f);

	if (fallbackFace.area() > 0) {
		width = std::max(width, fallbackFace.width * 0.95f);
		height = std::max(height, fallbackFace.height * 0.95f);
	}

	const float centerX = eyeCenter.x;
	const float centerY = eyeCenter.y + height * 0.18f;

	return ClampRect(
		cv::Rect(
			cvRound(centerX - width * 0.5f),
			cvRound(centerY - height * 0.5f),
			cvRound(width),
			cvRound(height)),
		imageSize);
}

static bool SelectFaceDetection(
	const std::vector<FaceDetectionResult>& detections,
	const cv::Rect& currentTrackArea,
	bool preferCurrentFace,
	FaceDetectionResult& selectedFace)
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

	selectedFace = detections[selectedIndex];
	return true;
}

class MediaPipeFaceMeshDetector : public FaceDetectionBackend {
public:
	static std::unique_ptr<FaceDetectionBackend> Create(wxString& scriptPath, wxString& error)
	{
		const wxString backendScript = FindFirstExistingFile(
			GetPackagedFileCandidates(wxT("mediapipe_face_mesh_backend.py")));
		if (backendScript.IsEmpty()) {
			error = wxT("Could not find mediapipe_face_mesh_backend.py");
			return std::unique_ptr<FaceDetectionBackend>();
		}

		std::vector<wxString> commandCandidates;
#if defined(__WXMSW__)
		commandCandidates.push_back(wxString::Format(wxT("python \"%s\""), backendScript.c_str()));
		commandCandidates.push_back(wxString::Format(wxT("py -3 \"%s\""), backendScript.c_str()));
#else
		commandCandidates.push_back(wxString::Format(wxT("python3 \"%s\""), backendScript.c_str()));
		commandCandidates.push_back(wxString::Format(wxT("python \"%s\""), backendScript.c_str()));
#endif

		wxString lastError = wxT("Could not start Python MediaPipe backend");
		for (size_t i = 0; i < commandCandidates.size(); ++i) {
			std::unique_ptr<MediaPipeFaceMeshDetector> detector(new MediaPipeFaceMeshDetector());
			if (detector->Start(commandCandidates[i], lastError)) {
				scriptPath = backendScript;
				error.clear();
				return std::unique_ptr<FaceDetectionBackend>(detector.release());
			}
		}

		error = lastError;
		return std::unique_ptr<FaceDetectionBackend>();
	}

	virtual ~MediaPipeFaceMeshDetector()
	{
		Stop();
	}

	virtual const char* GetName() const { return "MediaPipe Face Detection"; }

	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		FaceDetectionResult& selectedFace)
	{
		wxUnusedVar(currentTrackArea);
		wxUnusedVar(preferCurrentFace);

		if (image.empty()) {
			return false;
		}
		if (!EnsureBackendRunning()) return false;

		std::vector<uchar> encodedFrame;
		if (!cv::imencode(".png", image, encodedFrame)) {
			return false;
		}

		const uint32_t frameSize = static_cast<uint32_t>(encodedFrame.size());
		if (m_transportSocket == NULL ||
			!SocketSendMessage(*m_transportSocket, encodedFrame.data(), encodedFrame.size())) {
			SLOG_WARNING(
				"MediaPipe backend socket write failed: frameSize=%u lastCount=%llu connected=%d",
				frameSize,
				m_transportSocket != NULL
					? static_cast<unsigned long long>(m_transportSocket->LastCount())
					: 0ULL,
				m_transportSocket != NULL && m_transportSocket->IsConnected() ? 1 : 0);
			DrainErrorStream();
			Stop();
			return false;
		}

		wxString responseLine;
		if (m_transportSocket == NULL || !SocketReceiveMessage(*m_transportSocket, responseLine)) {
			SLOG_WARNING("MediaPipe backend socket response timed out or failed");
			DrainErrorStream();
			Stop();
			return false;
		}
		SLOG_DEBUG("MediaPipe backend response: %s", ToUtf8(responseLine).c_str());
		DrainErrorStream();

		return ParseResponse(responseLine, 1.0, image.size(), selectedFace);
	}

private:
	MediaPipeFaceMeshDetector()
		: m_process(NULL)
		, m_pid(0)
		, m_childStdout(NULL)
		, m_childStderr(NULL)
		, m_transportSocket(NULL)
	{
	}

	bool Start(const wxString& command, wxString& error)
	{
		m_command = command;
		Stop();

		m_process = new wxProcess();
		m_process->Redirect();

		int executeFlags = wxEXEC_ASYNC;
#if defined(__WXMSW__) && defined(wxEXEC_HIDE_CONSOLE)
		executeFlags |= wxEXEC_HIDE_CONSOLE;
#endif
		m_pid = wxExecute(command, executeFlags, m_process);
		if (m_pid == 0) {
			delete m_process;
			m_process = NULL;
			error = wxString::Format(wxT("Failed to execute: %s"), command.c_str());
			return false;
		}

		m_childStdout = m_process->GetInputStream();
		m_childStderr = m_process->GetErrorStream();

		if (m_childStdout == NULL) {
			error = wxT("MediaPipe backend process streams are unavailable");
			Stop();
			return false;
		}

		wxString readyLine;
		if (!ReadLineWithTimeout(readyLine, MEDIAPIPE_READY_TIMEOUT_MS)) {
			DrainErrorStream();
			error = wxString::Format(
				wxT("MediaPipe backend did not become ready (response: %s)"),
				readyLine.c_str());
			Stop();
			return false;
		}
		if (!ConnectTransportSocket(readyLine, error)) {
			DrainErrorStream();
			Stop();
			return false;
		}

		if (m_process->GetOutputStream() != NULL) {
			m_process->CloseOutput();
		}

		DrainErrorStream();
		return true;
	}

	void Stop()
	{
		if (m_transportSocket != NULL) {
			if (m_transportSocket->IsConnected()) {
				m_transportSocket->Close();
			}
			delete m_transportSocket;
			m_transportSocket = NULL;
		}

		if (m_process != NULL) {
			if (m_process->GetOutputStream() != NULL) {
				m_process->CloseOutput();
			}
			if (m_pid != 0) {
				const wxKillError killResult =
					wxProcess::Kill(m_pid, wxSIGTERM, NULL, wxKILL_CHILDREN);
				if (killResult != wxKILL_OK && killResult != wxKILL_NO_PROCESS) {
					SLOG_WARNING("Failed to terminate MediaPipe backend process: pid=%ld", m_pid);
				}
			}
			delete m_process;
			m_process = NULL;
		}

		m_pid = 0;
		m_childStdout = NULL;
		m_childStderr = NULL;
	}

	bool EnsureBackendRunning()
	{
		if (m_process != NULL &&
			m_transportSocket != NULL &&
			m_transportSocket->IsConnected()) {
			return true;
		}

		if (m_command.IsEmpty()) return false;

		wxString error;
		if (!Start(m_command, error)) {
			SLOG_WARNING(
				"MediaPipe backend restart failed: %s",
				ToUtf8(error).c_str());
			return false;
		}

		SLOG_INFO("MediaPipe backend restarted");
		return true;
	}

	bool ReadLineWithTimeout(wxString& line, long timeoutMs)
	{
		line.clear();
		if (m_childStdout == NULL) return false;

		wxStopWatch timer;
		while (timer.Time() < timeoutMs) {
			if (m_childStdout->CanRead()) {
				wxTextInputStream textInput(*m_childStdout);
				line = textInput.ReadLine();
				return true;
			}
			wxMilliSleep(10);
		}

		return false;
	}

	bool ConnectTransportSocket(const wxString& readyLine, wxString& error)
	{
		const wxString prefix = wxT("READY PORT=");
		if (readyLine.Left(prefix.Length()) != prefix) {
			error = wxString::Format(
				wxT("Unexpected MediaPipe backend ready response: %s"),
				readyLine.c_str());
			return false;
		}

		wxString portText = readyLine.Mid(prefix.Length());
		portText.Trim(true);
		portText.Trim(false);

		long portLong = 0;
		if (!portText.ToLong(&portLong) || portLong <= 0 || portLong > 65535) {
			error = wxString::Format(
				wxT("Invalid MediaPipe backend port: %s"),
				portText.c_str());
			return false;
		}

		std::unique_ptr<wxSocketClient> socket(new wxSocketClient());
		socket->SetFlags(wxSOCKET_BLOCK | wxSOCKET_WAITALL);
		socket->SetTimeout(TimeoutMsToSeconds(MEDIAPIPE_SOCKET_CONNECT_TIMEOUT_MS));

		wxIPV4address address;
		address.Hostname(wxT("127.0.0.1"));
		address.Service(static_cast<unsigned short>(portLong));

		if (!socket->Connect(address, true)) {
			error = wxString::Format(
				wxT("Failed to connect to MediaPipe backend transport socket on port %ld"),
				portLong);
			return false;
		}
		socket->SetTimeout(TimeoutMsToSeconds(MEDIAPIPE_SOCKET_IO_TIMEOUT_MS));

		m_transportSocket = socket.release();
		return true;
	}

	void DrainErrorStream()
	{
		if (m_childStderr == NULL) return;

		while (m_childStderr->CanRead()) {
			wxTextInputStream textInput(*m_childStderr);
			const wxString errorLine = textInput.ReadLine();
			if (!errorLine.IsEmpty()) {
				SLOG_WARNING("MediaPipe backend: %s", ToUtf8(errorLine).c_str());
			}
		}
	}

	bool ParseResponse(
		const wxString& responseLine,
		double scale,
		const cv::Size& imageSize,
		FaceDetectionResult& selectedFace)
	{
		std::istringstream iss(ToUtf8(responseLine));
		iss.imbue(std::locale::classic());
		std::string status;
		iss >> status;
		if (status != "OK") return false;

		double x = 0.0;
		double y = 0.0;
		double width = 0.0;
		double height = 0.0;
		double score = 0.0;
		iss >> x >> y >> width >> height >> score;
		if (!iss) {
			SLOG_WARNING(
				"MediaPipe backend response parse failed after bbox: %s",
				ToUtf8(responseLine).c_str());
			return false;
		}

		selectedFace = FaceDetectionResult();
		selectedFace.box = ClampRect(
			cv::Rect(
				cvRound(x / scale),
				cvRound(y / scale),
				cvRound(width / scale),
				cvRound(height / scale)),
			imageSize);
		selectedFace.score = static_cast<float>(score);

		for (int i = 0; i < 5; ++i) {
			double landmarkX = 0.0;
			double landmarkY = 0.0;
			iss >> landmarkX >> landmarkY;
			if (!iss) {
				SLOG_WARNING(
					"MediaPipe backend response parse failed on landmark %d: %s",
					i,
					ToUtf8(responseLine).c_str());
				return false;
			}

			selectedFace.landmarks.push_back(Point2f(
				static_cast<float>(landmarkX / scale),
				static_cast<float>(landmarkY / scale)));
		}

		return selectedFace.box.area() > 0 && selectedFace.landmarks.size() >= 3;
	}

	wxProcess* m_process;
	long m_pid;
	wxInputStream* m_childStdout;
	wxInputStream* m_childStderr;
	wxSocketClient* m_transportSocket;
	wxString m_command;
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
		FaceDetectionResult& selectedFace)
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
		FaceDetectionResult& selectedFace)
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
			for (int landmarkIndex = 0; landmarkIndex < 5; ++landmarkIndex) {
				const float landmarkX = detections.at<float>(row, 4 + landmarkIndex * 2);
				const float landmarkY = detections.at<float>(row, 5 + landmarkIndex * 2);
				const Point2f landmark(
					static_cast<float>(landmarkX / scale),
					static_cast<float>(landmarkY / scale));
				if (face.contains(cv::Point(cvRound(landmark.x), cvRound(landmark.y)))) {
					result.landmarks.push_back(landmark);
				}
			}
			results.push_back(result);
		}

		return SelectFaceDetection(results, currentTrackArea, preferCurrentFace, selectedFace);
	}

private:
	YuNetFaceDetector() {}

	cv::Ptr<cv::FaceDetectorYN> m_detector;
};
#endif

static bool UsesSynchronousLandmarkTracking(const FaceDetectionBackend* backend)
{
	if (backend == NULL) return false;

	const std::string backendName = backend->GetName();
	// YuNet runs in-process and exposes richer landmarks, so it can drive the
	// tracker directly from the main loop.
	return backendName == "YuNet";
}

static bool UsesDetectionDrivenTracking(const FaceDetectionBackend* backend)
{
	if (backend == NULL) return false;

	return std::string(backend->GetName()) == "MediaPipe Face Detection";
}

static std::unique_ptr<FaceDetectionBackend> CreateFaceDetectionBackend(bool& available)
{
	available = false;

	wxString mediaPipeScript;
	wxString mediaPipeError;
	std::unique_ptr<FaceDetectionBackend> mediaPipeBackend =
		MediaPipeFaceMeshDetector::Create(mediaPipeScript, mediaPipeError);
	if (mediaPipeBackend.get() != NULL) {
		SLOG_INFO(
			"Using face detector backend: %s (%s)",
			mediaPipeBackend->GetName(),
			ToUtf8(mediaPipeScript).c_str());
		available = true;
		return mediaPipeBackend;
	}

	SLOG_WARNING(
		"MediaPipe Face Detection backend unavailable and fallback is disabled: %s",
		ToUtf8(mediaPipeError).c_str());
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

	m_useLandmarkTracking = UsesSynchronousLandmarkTracking(m_faceDetector.get());
	m_useDetectionDrivenTracking = UsesDetectionDrivenTracking(m_faceDetector.get());
	const bool useSynchronousTracking =
		m_useLandmarkTracking || m_useDetectionDrivenTracking;
	if (useSynchronousTracking) {
		SetCpuUsage(CVisionPipeline::CPU_HIGHEST);
		if (m_useDetectionDrivenTracking) {
			SLOG_INFO(
				"Using synchronous detection-driven tracking with %s",
				m_faceDetector->GetName());
		}
		else {
			SLOG_INFO(
				"Using synchronous landmark-driven tracking with %s",
				m_faceDetector->GetName());
		}
	}
	else {
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
			try {
				if (m_imgPrevColor.empty()) continue;

				{
					wxCriticalSectionLocker lock(m_imageCopyMutex);
					m_imgPrevColor.copyTo(m_imgThread);
				}

				ComputeFaceTrackArea(m_imgThread);
			}
			catch (const cv::Exception& e) {
				SLOG_WARNING("Face detection thread ignored OpenCV exception: %s", e.what());
			}
			catch (const std::exception& e) {
				SLOG_WARNING("Face detection thread ignored exception: %s", e.what());
			}
			catch (...) {
				SLOG_WARNING("Face detection thread ignored unknown exception");
			}
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
	const bool hadTrackedFace = IsFaceDetected();

	FaceDetectionResult detectedFace;
	if (m_faceDetector->Detect(image, currentTrackArea, IsFaceDetected(), detectedFace)) {
		if (!hadTrackedFace) {
			m_hasPreviousFaceAnchor = false;
		}
		m_faceLandmarks = detectedFace.landmarks;
		if (m_useLandmarkTracking && detectedFace.landmarks.size() >= 3) {
			m_faceLocation = BuildTrackAreaFromLandmarks(
				detectedFace.landmarks,
				detectedFace.box,
				image.size());
		}
		else {
			m_faceLocation = detectedFace.box;
		}
		m_faceLocationStatus = 1;

		m_waitTime.Reset();
		m_trackAreaTimeout.Reset();

		if (m_useDetectionDrivenTracking) {
			static unsigned long s_lastDetectionLog = 0;
			const unsigned long now = CTimeUtil::GetMiliCount();
			if (now - s_lastDetectionLog >= 1000) {
				SLOG_DEBUG(
					"MediaPipe detection accepted: box=(%d,%d %dx%d) landmarks=%d",
					m_faceLocation.x,
					m_faceLocation.y,
					m_faceLocation.width,
					m_faceLocation.height,
					(int) detectedFace.landmarks.size());
				s_lastDetectionLog = now;
			}
		}
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
	std::vector<Point2f> detectedLandmarks;

	xVel = 0;
	yVel = 0;

	// Face location has been updated?
	if (m_faceLocationStatus) {
		trackArea = m_faceLocation;
		detectedLandmarks = m_faceLandmarks;
		m_faceLandmarks.clear();
		m_faceLocationStatus = 0;
		updateFeatures = true;
	}
	else {
		cv::Rect box;
		m_trackArea.GetBoxImg(image, box);
		trackArea = box;
				
        // Need to update corners?
		if (m_corners.size() < MIN_ACTIVE_TRACK_CORNERS) updateFeatures = true;
	}	

	const cv::Rect trackAreaRoi = ClampRect(trackArea, image.size());
	if (trackAreaRoi.area() <= 0) {
		m_corners.clear();
		return;
	}

	if (m_useLandmarkTracking) {
		if (updateFeatures && detectedLandmarks.size() >= 3) {
			Point2f faceAnchor;
			if (GetLandmarkAnchor(detectedLandmarks, faceAnchor)) {
				if (m_hasPreviousFaceAnchor) {
					const float dx = m_previousFaceAnchor.x - faceAnchor.x;
					const float dy = m_previousFaceAnchor.y - faceAnchor.y;
					xVel = 2.0f * dx;
					yVel = 2.0f * -dy;
				}

				m_previousFaceAnchor = faceAnchor;
				m_hasPreviousFaceAnchor = true;
				m_corners = detectedLandmarks;

				const cv::Rect landmarkTrackArea = BuildTrackAreaFromLandmarks(
					detectedLandmarks,
					trackAreaRoi,
					image.size());
				m_trackArea.SetSizeImg(image, landmarkTrackArea.width, landmarkTrackArea.height);
				m_trackArea.SetCenterImg(
					image,
					cvRound(landmarkTrackArea.x + landmarkTrackArea.width / 2.0f),
					cvRound(landmarkTrackArea.y + landmarkTrackArea.height / 2.0f));
				DrawCorners(image, m_corners, cv::Scalar(0, 255, 0));
				return;
			}
		}

		m_corners.clear();
		xVel = 0;
		yVel = 0;
		return;
	}

	if (m_useDetectionDrivenTracking) {
		if (updateFeatures && detectedLandmarks.size() >= 3) {
			Point2f faceAnchor;
			if (GetLandmarkAnchor(detectedLandmarks, faceAnchor)) {
				if (m_hasPreviousFaceAnchor) {
					const float dx = m_previousFaceAnchor.x - faceAnchor.x;
					const float dy = m_previousFaceAnchor.y - faceAnchor.y;
					xVel = 2.0f * dx;
					yVel = 2.0f * -dy;
				}

				m_previousFaceAnchor = faceAnchor;
				m_hasPreviousFaceAnchor = true;
				m_corners = detectedLandmarks;

				const cv::Rect trackedFaceArea = BuildTrackAreaFromLandmarks(
					detectedLandmarks,
					trackAreaRoi,
					image.size());
				m_trackArea.SetSizeImg(image, trackedFaceArea.width, trackedFaceArea.height);
				m_trackArea.SetCenterImg(
					image,
					cvRound(trackedFaceArea.x + trackedFaceArea.width / 2.0f),
					cvRound(trackedFaceArea.y + trackedFaceArea.height / 2.0f));
				DrawCorners(image, m_corners, cv::Scalar(0, 255, 0));

				static unsigned long s_lastTrackingLog = 0;
				const unsigned long now = CTimeUtil::GetMiliCount();
				if (now - s_lastTrackingLog >= 1000) {
					SLOG_DEBUG(
						"MediaPipe tracking update: xVel=%.3f yVel=%.3f center=(%d,%d)",
						xVel,
						yVel,
						cvRound(trackedFaceArea.x + trackedFaceArea.width / 2.0f),
						cvRound(trackedFaceArea.y + trackedFaceArea.height / 2.0f));
					s_lastTrackingLog = now;
				}
			}
		}
		else {
			xVel = 0;
			yVel = 0;
		}
		return;
	}

	if (updateFeatures) {
		#define QUALITY_LEVEL  0.001   // 0.01
		#define MIN_DISTANTE 2

		CollectLandmarkCorners(m_imgPrev, detectedLandmarks, trackAreaRoi, m_corners);

		if (m_corners.size() > NUM_CORNERS) {
			m_corners.resize(NUM_CORNERS);
		}

		if (m_corners.empty()) {
			const cv::Rect featuresTrackArea = BuildFeatureTrackArea(trackAreaRoi, m_imgPrev.size());
			if (featuresTrackArea.area() <= 0) {
				m_corners.clear();
				return;
			}

			cv::Mat prevImg = m_imgPrev(featuresTrackArea);

			goodFeaturesToTrack(prevImg, m_corners, NUM_CORNERS, QUALITY_LEVEL, MIN_DISTANTE);

			if (m_corners.empty() && featuresTrackArea != trackAreaRoi) {
				prevImg = m_imgPrev(trackAreaRoi);
				goodFeaturesToTrack(prevImg, m_corners, NUM_CORNERS, QUALITY_LEVEL, MIN_DISTANTE);
				if (!m_corners.empty()) {
					RefineCornersInRoi(prevImg, m_corners);
					OffsetCorners(m_corners, trackAreaRoi.x, trackAreaRoi.y);
					FilterLowerFaceCorners(m_corners, trackAreaRoi);
				}
			}
			else {
				RefineCornersInRoi(prevImg, m_corners);
				OffsetCorners(m_corners, featuresTrackArea.x, featuresTrackArea.y);
				FilterLowerFaceCorners(m_corners, trackAreaRoi);
			}
		}
	}

	if (slog_get_priority() >= SLOG_PRIO_DEBUG) {
	    DrawCorners(image, m_corners, cv::Scalar(255, 0, 0));
    }

	//
	// Track corners
	//
    cv::Mat prevImg = m_imgPrev(trackAreaRoi);
    cv::Mat currImg = m_imgCurr(trackAreaRoi);

	// Update corners location for the new ROI
	for (int i = 0; i < m_corners.size(); i++) {
		m_corners[i].x -= trackAreaRoi.x;
		m_corners[i].y -= trackAreaRoi.y;
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
			m_corners[i].x < trackAreaRoi.width &&
			m_corners[i].y >= 0 &&
			m_corners[i].y < trackAreaRoi.height) {
			dx += m_corners[i].x - new_corners[i].x;
			dy += m_corners[i].y - new_corners[i].y;

			// Update corner location, relative to full `image`
			new_corners[i].x += trackAreaRoi.x;
			new_corners[i].y += trackAreaRoi.y;

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
		xVel = 0;
		yVel = 0;
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
		{
			wxCriticalSectionLocker lock(m_imageCopyMutex);

			if (m_useLandmarkTracking || m_useDetectionDrivenTracking) {
				ComputeFaceTrackArea(detectFrame);
			}

			NewTracker(image, xVel, yVel);

			// Store current image as previous
			cv::swap(m_imgPrev, m_imgCurr);
			m_imgPrevColor = detectFrame;
		}

		// Notifies face detection thread when needed
		if (!m_useLandmarkTracking &&
			!m_useDetectionDrivenTracking &&
			m_trackFace &&
			m_faceDetectionAvailable) {
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
	catch (const cv::Exception& e) {
		SLOG_ERR("OpenCV exception in ProcessImage: %s", e.what());
		xVel = 0;
		yVel = 0;
		return true;
	}
	catch (const std::exception& e) {
		SLOG_ERR("Exception in ProcessImage: %s", e.what());
		xVel = 0;
		yVel = 0;
		return true;
	}
	catch (...) {
		SLOG_ERR("Unknown exception in ProcessImage");
		xVel = 0;
		yVel = 0;
		return true;
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
	m_useLandmarkTracking = false;
	m_useDetectionDrivenTracking = false;
	m_hasPreviousFaceAnchor = false;
	m_previousFaceAnchor = Point2f(0.0f, 0.0f);
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
