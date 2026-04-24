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
#include <opencv2/calib3d.hpp>
#if defined(ENABLE_ONNXRUNTIME_BACKEND)
#include <onnxruntime_cxx_api.h>
#endif

#include <math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <wx/filename.h>
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

enum { MIN_ACTIVE_TRACK_CORNERS = 6 };

struct FaceDetectionResult {
	cv::Rect box;
	float score;
	std::vector<cv::Point2f> landmarks;
	bool hasPose;
	cv::Matx44f pose;

	FaceDetectionResult()
	: box(), score(0.0f), landmarks(), hasPose(false), pose(cv::Matx44f::eye())
	{}
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

const float DETECTION_DRIVEN_FACE_ANCHOR_Y = 0.42f;
const float DETECTION_DRIVEN_FACE_SMOOTHING = 0.20f;
const float DETECTION_DRIVEN_FACE_DEADZONE_PX = 1.5f;
const cv::Size HAAR_MIN_FACE_SIZE(65, 65);

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

static cv::Point2f GetDetectionDrivenFaceAnchor(const cv::Rect& faceBox)
{
	return cv::Point2f(
		faceBox.x + faceBox.width * 0.5f,
		faceBox.y + faceBox.height * DETECTION_DRIVEN_FACE_ANCHOR_Y);
}

static cv::Point2f SmoothPoint(
	const cv::Point2f& previous,
	const cv::Point2f& current,
	float alpha)
{
	return previous * (1.0f - alpha) + current * alpha;
}

static void ApplyDeadzone(float& dx, float& dy, float threshold)
{
	if (fabs(dx) < threshold) dx = 0.0f;
	if (fabs(dy) < threshold) dy = 0.0f;
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

#if defined(ENABLE_ONNXRUNTIME_BACKEND)

class OnnxBlazeFaceDetector : public FaceDetectionBackend {
public:
	static std::unique_ptr<FaceDetectionBackend> Create(wxString& modelPath, wxString& error)
	{
		const wxString modelFile = FindFirstExistingFile(
			GetPackagedFileCandidates(wxT("face_detection_back_256x256.onnx")));
		if (modelFile.IsEmpty()) {
			error = wxT("Could not find face_detection_back_256x256.onnx");
			return std::unique_ptr<FaceDetectionBackend>();
		}

		try {
			std::unique_ptr<OnnxBlazeFaceDetector> detector(new OnnxBlazeFaceDetector());
			if (!detector->Initialize(modelFile, error)) {
				return std::unique_ptr<FaceDetectionBackend>();
			}
			modelPath = modelFile;
			error.clear();
			return std::unique_ptr<FaceDetectionBackend>(detector.release());
		}
		catch (const Ort::Exception& e) {
			error = wxString::Format(
				wxT("BlazeFace init failed: %s"),
				wxString::FromUTF8(e.what()).c_str());
		}
		catch (const std::exception& e) {
			error = wxString::Format(
				wxT("BlazeFace init failed: %s"),
				wxString::FromUTF8(e.what()).c_str());
		}
		catch (...) {
			error = wxT("BlazeFace init failed (unknown exception)");
		}

		return std::unique_ptr<FaceDetectionBackend>();
	}

	virtual ~OnnxBlazeFaceDetector() {}

	virtual const char* GetName() const { return "MediaPipe BlazeFace"; }

	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		FaceDetectionResult& selectedFace)
	{
		if (image.empty() || m_session.get() == NULL) return false;

		const int imgW = image.cols;
		const int imgH = image.rows;
		const float scale = std::min(
			static_cast<float>(kInputSize) / static_cast<float>(imgW),
			static_cast<float>(kInputSize) / static_cast<float>(imgH));
		const int newW = std::max(1, cvRound(imgW * scale));
		const int newH = std::max(1, cvRound(imgH * scale));
		const int padX = (kInputSize - newW) / 2;
		const int padY = (kInputSize - newH) / 2;

		cv::Mat resized;
		cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

		cv::Mat rgb;
		if (resized.channels() == 1) {
			cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
		}
		else if (resized.channels() == 4) {
			cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
		}
		else {
			cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
		}

		cv::Mat padded(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
		rgb.copyTo(padded(cv::Rect(padX, padY, newW, newH)));

		cv::Mat normalized;
		padded.convertTo(normalized, CV_32FC3, 1.0 / 127.5, -1.0);
		if (!normalized.isContinuous()) {
			normalized = normalized.clone();
		}

		std::array<int64_t, 4> inputShape = { 1, kInputSize, kInputSize, 3 };
		const size_t inputElements =
			static_cast<size_t>(kInputSize) * kInputSize * 3;

		try {
			Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
				OrtArenaAllocator, OrtMemTypeDefault);
			Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
				memInfo,
				reinterpret_cast<float*>(normalized.data),
				inputElements,
				inputShape.data(),
				inputShape.size());

			const char* inputNames[1] = { m_inputName.c_str() };
			std::vector<const char*> outputNames;
			outputNames.reserve(m_outputNames.size());
			for (size_t i = 0; i < m_outputNames.size(); ++i) {
				outputNames.push_back(m_outputNames[i].c_str());
			}

			std::vector<Ort::Value> outputs = m_session->Run(
				Ort::RunOptions{ nullptr },
				inputNames,
				&inputTensor,
				1,
				outputNames.data(),
				outputNames.size());

			return DecodeAndSelect(
				outputs, scale, padX, padY, image.size(),
				currentTrackArea, preferCurrentFace, selectedFace);
		}
		catch (const Ort::Exception& e) {
			SLOG_WARNING("BlazeFace inference failed: %s", e.what());
		}
		catch (const std::exception& e) {
			SLOG_WARNING("BlazeFace inference failed: %s", e.what());
		}
		catch (...) {
			SLOG_WARNING("BlazeFace inference failed (unknown exception)");
		}

		return false;
	}

private:
	struct Anchor {
		float x;
		float y;
	};

	static const int kInputSize = 256;
	static const int kNumAnchors = 896;
	static const int kNumHeads = 2;
	static const int kRegressorSize = 16;
	static const int kLandmarkCount = 5;
	static constexpr float kScoreThreshold = 0.5f;

	OnnxBlazeFaceDetector()
	: m_env(ORT_LOGGING_LEVEL_WARNING, "eviacam_blazeface")
	{}

	bool Initialize(const wxString& modelPath, wxString& error)
	{
		Ort::SessionOptions options;
		options.SetIntraOpNumThreads(1);
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#if defined(__WXMSW__)
		const std::wstring widePath = modelPath.ToStdWstring();
		m_session.reset(new Ort::Session(m_env, widePath.c_str(), options));
#else
		const std::string utf8Path = ToUtf8(modelPath);
		m_session.reset(new Ort::Session(m_env, utf8Path.c_str(), options));
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		{
			Ort::AllocatedStringPtr namePtr =
				m_session->GetInputNameAllocated(0, allocator);
			m_inputName = namePtr.get();
		}

		const size_t outputCount = m_session->GetOutputCount();
		m_outputNames.reserve(outputCount);
		for (size_t i = 0; i < outputCount; ++i) {
			Ort::AllocatedStringPtr namePtr =
				m_session->GetOutputNameAllocated(i, allocator);
			m_outputNames.push_back(std::string(namePtr.get()));
		}

		if (outputCount != 4) {
			error = wxString::Format(
				wxT("Unexpected BlazeFace output count: %u"),
				static_cast<unsigned>(outputCount));
			return false;
		}

		GenerateAnchors();
		if (m_anchors.size() != kNumAnchors) {
			error = wxString::Format(
				wxT("Anchor generation produced %u entries (expected %d)"),
				static_cast<unsigned>(m_anchors.size()),
				kNumAnchors);
			return false;
		}

		return true;
	}

	void GenerateAnchors()
	{
		m_anchors.clear();
		m_anchors.reserve(kNumAnchors);

		const int numLayers = 4;
		const int strides[4] = { 16, 32, 32, 32 };
		const float anchorOffsetX = 0.5f;
		const float anchorOffsetY = 0.5f;

		int layer = 0;
		while (layer < numLayers) {
			int lastSameStrideLayer = layer;
			int anchorsPerCell = 0;
			while (lastSameStrideLayer < numLayers &&
				strides[lastSameStrideLayer] == strides[layer]) {
				anchorsPerCell += 2; // base aspect-ratio anchor + interpolated anchor
				lastSameStrideLayer += 1;
			}

			const int stride = strides[layer];
			const int fmH = kInputSize / stride;
			const int fmW = kInputSize / stride;
			for (int y = 0; y < fmH; ++y) {
				for (int x = 0; x < fmW; ++x) {
					for (int c = 0; c < anchorsPerCell; ++c) {
						Anchor a;
						a.x = (static_cast<float>(x) + anchorOffsetX) /
							static_cast<float>(fmW);
						a.y = (static_cast<float>(y) + anchorOffsetY) /
							static_cast<float>(fmH);
						m_anchors.push_back(a);
					}
				}
			}
			layer = lastSameStrideLayer;
		}
	}

	bool DecodeAndSelect(
		const std::vector<Ort::Value>& outputs,
		float scale,
		int padX,
		int padY,
		const cv::Size& imageSize,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		FaceDetectionResult& selectedFace)
	{
		const float* classifHeads[kNumHeads] = { NULL, NULL };
		const float* regressHeads[kNumHeads] = { NULL, NULL };
		int classCounts[kNumHeads] = { 0, 0 };
		int classIndex = 0;
		int regIndex = 0;

		for (size_t i = 0; i < outputs.size(); ++i) {
			const Ort::Value& out = outputs[i];
			auto info = out.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> shape = info.GetShape();
			if (shape.size() != 3) continue;
			const int64_t anchors = shape[1];
			const int64_t width = shape[2];
			const float* data = out.GetTensorData<float>();
			if (width == 1 && classIndex < kNumHeads) {
				classifHeads[classIndex] = data;
				classCounts[classIndex] = static_cast<int>(anchors);
				classIndex += 1;
			}
			else if (width == kRegressorSize && regIndex < kNumHeads) {
				regressHeads[regIndex] = data;
				regIndex += 1;
			}
		}

		if (classIndex != kNumHeads || regIndex != kNumHeads) {
			SLOG_WARNING(
				"BlazeFace output layout unexpected: class=%d reg=%d",
				classIndex, regIndex);
			return false;
		}

		const float inputSizeF = static_cast<float>(kInputSize);
		std::vector<FaceDetectionResult> detections;
		int anchorOffset = 0;
		for (int h = 0; h < kNumHeads; ++h) {
			const int count = classCounts[h];
			const float* classif = classifHeads[h];
			const float* reg = regressHeads[h];
			for (int i = 0; i < count; ++i) {
				if (anchorOffset >= static_cast<int>(m_anchors.size())) break;
				float rawScore = classif[i];
				if (rawScore < -100.0f) rawScore = -100.0f;
				if (rawScore > 100.0f) rawScore = 100.0f;
				const float score = 1.0f / (1.0f + std::exp(-rawScore));
				if (score < kScoreThreshold) {
					anchorOffset += 1;
					continue;
				}

				const float* r = reg + i * kRegressorSize;
				const Anchor& anchor = m_anchors[anchorOffset];
				const float cx = r[0] / inputSizeF + anchor.x;
				const float cy = r[1] / inputSizeF + anchor.y;
				const float w = r[2] / inputSizeF;
				const float hh = r[3] / inputSizeF;

				const float boxX = (cx - w * 0.5f) * inputSizeF;
				const float boxY = (cy - hh * 0.5f) * inputSizeF;
				const float boxW = w * inputSizeF;
				const float boxH = hh * inputSizeF;

				const float origX = (boxX - padX) / scale;
				const float origY = (boxY - padY) / scale;
				const float origW = boxW / scale;
				const float origH = boxH / scale;

				cv::Rect rect(
					cvRound(origX),
					cvRound(origY),
					cvRound(origW),
					cvRound(origH));
				rect = ClampRect(rect, imageSize);
				if (rect.area() <= 0) {
					anchorOffset += 1;
					continue;
				}

				FaceDetectionResult result;
				result.box = rect;
				result.score = score;
				for (int k = 0; k < kLandmarkCount; ++k) {
					const float kpX =
						(r[4 + 2 * k] / inputSizeF + anchor.x) * inputSizeF;
					const float kpY =
						(r[5 + 2 * k] / inputSizeF + anchor.y) * inputSizeF;
					const float origKpX = (kpX - padX) / scale;
					const float origKpY = (kpY - padY) / scale;
					result.landmarks.push_back(cv::Point2f(origKpX, origKpY));
				}
				detections.push_back(result);
				anchorOffset += 1;
			}
		}

		return SelectFaceDetection(
			detections, currentTrackArea, preferCurrentFace, selectedFace);
	}

	Ort::Env m_env;
	std::unique_ptr<Ort::Session> m_session;
	std::string m_inputName;
	std::vector<std::string> m_outputNames;
	std::vector<Anchor> m_anchors;
};

class OnnxFaceLandmarkDetector {
public:
	static const int kInputSize = 192;
	static const int kNumLandmarks = 468;

	static std::unique_ptr<OnnxFaceLandmarkDetector> Create(
		wxString& modelPath, wxString& error)
	{
		const wxString modelFile = FindFirstExistingFile(
			GetPackagedFileCandidates(wxT("face_mesh_192x192.onnx")));
		if (modelFile.IsEmpty()) {
			error = wxT("Could not find face_mesh_192x192.onnx");
			return std::unique_ptr<OnnxFaceLandmarkDetector>();
		}

		try {
			std::unique_ptr<OnnxFaceLandmarkDetector> detector(
				new OnnxFaceLandmarkDetector());
			if (!detector->Initialize(modelFile, error)) {
				return std::unique_ptr<OnnxFaceLandmarkDetector>();
			}
			modelPath = modelFile;
			error.clear();
			return detector;
		}
		catch (const Ort::Exception& e) {
			error = wxString::Format(
				wxT("FaceMesh init failed: %s"),
				wxString::FromUTF8(e.what()).c_str());
		}
		catch (const std::exception& e) {
			error = wxString::Format(
				wxT("FaceMesh init failed: %s"),
				wxString::FromUTF8(e.what()).c_str());
		}
		catch (...) {
			error = wxT("FaceMesh init failed (unknown exception)");
		}

		return std::unique_ptr<OnnxFaceLandmarkDetector>();
	}

	~OnnxFaceLandmarkDetector() {}

	bool Run(
		const cv::Mat& image,
		const cv::Rect& faceBox,
		std::vector<cv::Point3f>& landmarks,
		cv::Matx44f& pose,
		bool& hasPose,
		float& score)
	{
		landmarks.clear();
		hasPose = false;
		pose = cv::Matx44f::eye();
		score = 0.0f;

		if (image.empty() || m_session.get() == NULL) return false;
		if (faceBox.width <= 0 || faceBox.height <= 0) return false;

		cv::Rect expanded = ExpandToSquare(faceBox, image.size(), 1.5f);
		if (expanded.width <= 0 || expanded.height <= 0) return false;

		cv::Mat crop;
		try {
			crop = image(expanded).clone();
		}
		catch (const cv::Exception& e) {
			SLOG_WARNING("FaceMesh crop failed: %s", e.what());
			return false;
		}

		cv::Mat resized;
		cv::resize(crop, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);

		cv::Mat rgb;
		if (resized.channels() == 1) {
			cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
		}
		else if (resized.channels() == 4) {
			cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
		}
		else {
			cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
		}

		cv::Mat normalized;
		rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0, 0.0);

		const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
		std::vector<float> chwBuffer(plane * 3);
		std::vector<cv::Mat> channels(3);
		for (int c = 0; c < 3; ++c) {
			channels[c] = cv::Mat(
				kInputSize, kInputSize, CV_32FC1,
				chwBuffer.data() + c * plane);
		}
		cv::split(normalized, channels);

		std::array<int64_t, 4> inputShape = { 1, 3, kInputSize, kInputSize };

		try {
			Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
				OrtArenaAllocator, OrtMemTypeDefault);
			Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
				memInfo,
				chwBuffer.data(),
				chwBuffer.size(),
				inputShape.data(),
				inputShape.size());

			const char* inputNames[1] = { m_inputName.c_str() };
			std::vector<const char*> outputNames;
			outputNames.reserve(m_outputNames.size());
			for (size_t i = 0; i < m_outputNames.size(); ++i) {
				outputNames.push_back(m_outputNames[i].c_str());
			}

			std::vector<Ort::Value> outputs = m_session->Run(
				Ort::RunOptions{ nullptr },
				inputNames,
				&inputTensor,
				1,
				outputNames.data(),
				outputNames.size());

			return Decode(outputs, expanded, landmarks, pose, hasPose, score);
		}
		catch (const Ort::Exception& e) {
			SLOG_WARNING("FaceMesh inference failed: %s", e.what());
		}
		catch (const std::exception& e) {
			SLOG_WARNING("FaceMesh inference failed: %s", e.what());
		}
		catch (...) {
			SLOG_WARNING("FaceMesh inference failed (unknown exception)");
		}

		return false;
	}

private:
	OnnxFaceLandmarkDetector()
	: m_env(ORT_LOGGING_LEVEL_WARNING, "eviacam_facemesh")
	{}

	bool Initialize(const wxString& modelPath, wxString& error)
	{
		Ort::SessionOptions options;
		options.SetIntraOpNumThreads(1);
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#if defined(__WXMSW__)
		const std::wstring widePath = modelPath.ToStdWstring();
		m_session.reset(new Ort::Session(m_env, widePath.c_str(), options));
#else
		const std::string utf8Path = ToUtf8(modelPath);
		m_session.reset(new Ort::Session(m_env, utf8Path.c_str(), options));
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		{
			Ort::AllocatedStringPtr namePtr =
				m_session->GetInputNameAllocated(0, allocator);
			m_inputName = namePtr.get();
		}

		const size_t outputCount = m_session->GetOutputCount();
		m_outputNames.reserve(outputCount);
		for (size_t i = 0; i < outputCount; ++i) {
			Ort::AllocatedStringPtr namePtr =
				m_session->GetOutputNameAllocated(i, allocator);
			m_outputNames.push_back(std::string(namePtr.get()));
		}

		if (outputCount < 2) {
			error = wxString::Format(
				wxT("Unexpected FaceMesh output count: %u"),
				static_cast<unsigned>(outputCount));
			return false;
		}

		return true;
	}

	static cv::Rect ExpandToSquare(
		const cv::Rect& box, const cv::Size& imageSize, float scale)
	{
		const float cx = box.x + box.width * 0.5f;
		const float cy = box.y + box.height * 0.5f;
		const float side = std::max(box.width, box.height) * scale;
		const float half = side * 0.5f;
		cv::Rect expanded(
			cvRound(cx - half),
			cvRound(cy - half),
			cvRound(side),
			cvRound(side));
		return ClampRect(expanded, imageSize);
	}

	bool Decode(
		const std::vector<Ort::Value>& outputs,
		const cv::Rect& cropRect,
		std::vector<cv::Point3f>& landmarks,
		cv::Matx44f& pose,
		bool& hasPose,
		float& score)
	{
		const float* landmarkData = NULL;
		const float* scoreData = NULL;

		for (size_t i = 0; i < outputs.size(); ++i) {
			const Ort::Value& out = outputs[i];
			auto info = out.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> shape = info.GetShape();
			int64_t total = 1;
			for (size_t k = 0; k < shape.size(); ++k) {
				total *= shape[k] > 0 ? shape[k] : 1;
			}
			const float* data = out.GetTensorData<float>();
			if (total == kNumLandmarks * 3) {
				landmarkData = data;
			}
			else if (total == 1) {
				scoreData = data;
			}
		}

		if (landmarkData == NULL) {
			SLOG_WARNING("FaceMesh: could not locate landmark output");
			return false;
		}

		score = (scoreData != NULL) ? *scoreData : 0.0f;

		const float sx = static_cast<float>(cropRect.width) /
			static_cast<float>(kInputSize);
		const float sy = static_cast<float>(cropRect.height) /
			static_cast<float>(kInputSize);

		landmarks.reserve(kNumLandmarks);
		for (int i = 0; i < kNumLandmarks; ++i) {
			const float lx = landmarkData[i * 3 + 0] * sx + cropRect.x;
			const float ly = landmarkData[i * 3 + 1] * sy + cropRect.y;
			const float lz = landmarkData[i * 3 + 2] *
				0.5f * (sx + sy);
			landmarks.push_back(cv::Point3f(lx, ly, lz));
		}

		hasPose = SolveHeadPose(landmarks, cropRect, pose);
		return true;
	}

	static bool SolveHeadPose(
		const std::vector<cv::Point3f>& landmarks,
		const cv::Rect& cropRect,
		cv::Matx44f& pose)
	{
		// Canonical 3D anchor points (mm, nose tip origin). Indices follow
		// the MediaPipe face mesh convention.
		const int idxs[6] = {
			1,    // nose tip
			152,  // chin
			33,   // right eye outer corner (subject's right = image left)
			263,  // left eye outer corner
			61,   // right mouth corner
			291   // left mouth corner
		};
		const cv::Point3f model3d[6] = {
			cv::Point3f(0.0f,   0.0f,   0.0f),
			cv::Point3f(0.0f, -63.6f, -12.5f),
			cv::Point3f(-43.3f, 32.7f, -26.0f),
			cv::Point3f(43.3f,  32.7f, -26.0f),
			cv::Point3f(-28.9f, -28.9f, -24.1f),
			cv::Point3f(28.9f,  -28.9f, -24.1f)
		};

		std::vector<cv::Point3f> objectPoints(6);
		std::vector<cv::Point2f> imagePoints(6);
		for (int i = 0; i < 6; ++i) {
			if (idxs[i] >= static_cast<int>(landmarks.size())) return false;
			objectPoints[i] = model3d[i];
			imagePoints[i] = cv::Point2f(
				landmarks[idxs[i]].x, landmarks[idxs[i]].y);
		}

		// Approximate intrinsics: focal length ~= crop diagonal, principal
		// point at crop center. Good enough for a relative head pose.
		const double focal = std::hypot(
			static_cast<double>(cropRect.width),
			static_cast<double>(cropRect.height));
		const double cx = cropRect.x + cropRect.width * 0.5;
		const double cy = cropRect.y + cropRect.height * 0.5;
		cv::Matx33d cameraMatrix(
			focal, 0.0, cx,
			0.0, focal, cy,
			0.0, 0.0, 1.0);
		cv::Matx<double, 4, 1> distCoeffs(0.0, 0.0, 0.0, 0.0);

		cv::Vec3d rvec, tvec;
		bool ok = false;
		try {
			ok = cv::solvePnP(
				objectPoints, imagePoints,
				cameraMatrix, distCoeffs,
				rvec, tvec,
				false, cv::SOLVEPNP_ITERATIVE);
		}
		catch (const cv::Exception& e) {
			SLOG_WARNING("solvePnP failed: %s", e.what());
			return false;
		}

		if (!ok) return false;

		cv::Matx33d R;
		cv::Rodrigues(rvec, R);

		pose = cv::Matx44f::eye();
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c) {
				pose(r, c) = static_cast<float>(R(r, c));
			}
			pose(r, 3) = static_cast<float>(tvec[r]);
		}
		return true;
	}

	Ort::Env m_env;
	std::unique_ptr<Ort::Session> m_session;
	std::string m_inputName;
	std::vector<std::string> m_outputNames;
};

class OnnxMediaPipeDetector : public FaceDetectionBackend {
public:
	static std::unique_ptr<FaceDetectionBackend> Create(
		std::unique_ptr<FaceDetectionBackend> innerDetector,
		std::unique_ptr<OnnxFaceLandmarkDetector> landmarker)
	{
		std::unique_ptr<OnnxMediaPipeDetector> composite(
			new OnnxMediaPipeDetector(
				std::move(innerDetector), std::move(landmarker)));
		return std::unique_ptr<FaceDetectionBackend>(composite.release());
	}

	virtual const char* GetName() const { return "MediaPipe BlazeFace + FaceMesh"; }

	virtual bool Detect(
		const cv::Mat& image,
		const cv::Rect& currentTrackArea,
		bool preferCurrentFace,
		FaceDetectionResult& selectedFace)
	{
		if (m_inner.get() == NULL) return false;

		FaceDetectionResult raw;
		if (!m_inner->Detect(image, currentTrackArea, preferCurrentFace, raw)) {
			return false;
		}

		selectedFace = raw;

		if (m_landmarker.get() != NULL) {
			std::vector<cv::Point3f> landmarks3d;
			cv::Matx44f pose;
			bool hasPose = false;
			float score = 0.0f;
			if (m_landmarker->Run(image, raw.box, landmarks3d, pose, hasPose, score)) {
				selectedFace.landmarks.clear();
				selectedFace.landmarks.reserve(landmarks3d.size());
				for (size_t i = 0; i < landmarks3d.size(); ++i) {
					selectedFace.landmarks.push_back(
						cv::Point2f(landmarks3d[i].x, landmarks3d[i].y));
				}
				selectedFace.hasPose = hasPose;
				selectedFace.pose = pose;
			}
		}

		return true;
	}

private:
	OnnxMediaPipeDetector(
		std::unique_ptr<FaceDetectionBackend> innerDetector,
		std::unique_ptr<OnnxFaceLandmarkDetector> landmarker)
	: m_inner(std::move(innerDetector))
	, m_landmarker(std::move(landmarker))
	{}

	std::unique_ptr<FaceDetectionBackend> m_inner;
	std::unique_ptr<OnnxFaceLandmarkDetector> m_landmarker;
};

#endif // ENABLE_ONNXRUNTIME_BACKEND

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

static bool UsesSynchronousLandmarkTracking(const FaceDetectionBackend* backend)
{
	wxUnusedVar(backend);
	return false;
}

static bool UsesDetectionDrivenTracking(const FaceDetectionBackend* backend)
{
	if (backend == NULL) return false;

	const std::string name(backend->GetName());
	return name == "MediaPipe BlazeFace" ||
		name == "MediaPipe BlazeFace + FaceMesh";
}

#if defined(ENABLE_ONNXRUNTIME_BACKEND)
static void LogOnnxRuntimeSmokeTest()
{
	static bool s_logged = false;
	if (s_logged) return;
	s_logged = true;

	try {
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "eviacam");
		std::string version = Ort::GetVersionString();
		SLOG_INFO(
			"onnxruntime smoke test OK: version=%s",
			version.empty() ? "unknown" : version.c_str());
	}
	catch (const Ort::Exception& e) {
		SLOG_WARNING("onnxruntime smoke test failed: %s", e.what());
	}
	catch (const std::exception& e) {
		SLOG_WARNING("onnxruntime smoke test failed: %s", e.what());
	}
	catch (...) {
		SLOG_WARNING("onnxruntime smoke test failed (unknown exception)");
	}
}
#endif

static std::unique_ptr<FaceDetectionBackend> CreateFaceDetectionBackend(bool& available)
{
	available = false;

#if defined(ENABLE_ONNXRUNTIME_BACKEND)
	LogOnnxRuntimeSmokeTest();

	wxString blazeFacePath;
	wxString blazeFaceError;
	std::unique_ptr<FaceDetectionBackend> blazeFaceBackend =
		OnnxBlazeFaceDetector::Create(blazeFacePath, blazeFaceError);
	if (blazeFaceBackend.get() != NULL) {
		wxString faceMeshPath;
		wxString faceMeshError;
		std::unique_ptr<OnnxFaceLandmarkDetector> landmarker =
			OnnxFaceLandmarkDetector::Create(faceMeshPath, faceMeshError);
		if (landmarker.get() != NULL) {
			std::unique_ptr<FaceDetectionBackend> composite =
				OnnxMediaPipeDetector::Create(
					std::move(blazeFaceBackend),
					std::move(landmarker));
			SLOG_INFO(
				"Using face detector backend: %s (detector=%s, landmarker=%s)",
				composite->GetName(),
				ToUtf8(blazeFacePath).c_str(),
				ToUtf8(faceMeshPath).c_str());
			available = true;
			return composite;
		}

		SLOG_WARNING(
			"FaceMesh landmarker unavailable, using BlazeFace only: %s",
			ToUtf8(faceMeshError).c_str());
		SLOG_INFO(
			"Using face detector backend: %s (%s)",
			blazeFaceBackend->GetName(),
			ToUtf8(blazeFacePath).c_str());
		available = true;
		return blazeFaceBackend;
	}

	SLOG_WARNING(
		"BlazeFace backend unavailable, falling back to Haar: %s",
		ToUtf8(blazeFaceError).c_str());
#endif

	wxString haarPath;
	wxString haarError;
	std::unique_ptr<FaceDetectionBackend> haarBackend =
		HaarFaceDetector::Create(haarPath, haarError);
	if (haarBackend.get() != NULL) {
		SLOG_INFO(
			"Using face detector backend: %s (%s)",
			haarBackend->GetName(),
			ToUtf8(haarPath).c_str());
		available = true;
		return haarBackend;
	}

	SLOG_WARNING(
		"Haar face detector backend unavailable: %s",
		ToUtf8(haarError).c_str());
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
				if (detectedFace.hasPose) {
					SLOG_DEBUG(
						"MediaPipe detection accepted: box=(%d,%d %dx%d) landmarks=%d pose=[tx=%.1f ty=%.1f tz=%.1f]",
						m_faceLocation.x,
						m_faceLocation.y,
						m_faceLocation.width,
						m_faceLocation.height,
						(int) detectedFace.landmarks.size(),
						detectedFace.pose(0, 3),
						detectedFace.pose(1, 3),
						detectedFace.pose(2, 3));
				}
				else {
					SLOG_DEBUG(
						"MediaPipe detection accepted: box=(%d,%d %dx%d) landmarks=%d",
						m_faceLocation.x,
						m_faceLocation.y,
						m_faceLocation.width,
						m_faceLocation.height,
						(int) detectedFace.landmarks.size());
				}
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
	bool hasFreshFaceLocation = false;
	std::vector<Point2f> detectedLandmarks;

	xVel = 0;
	yVel = 0;

	// Face location has been updated?
	if (m_faceLocationStatus) {
		trackArea = m_faceLocation;
		detectedLandmarks = m_faceLandmarks;
		m_faceLandmarks.clear();
		m_faceLocationStatus = 0;
		hasFreshFaceLocation = true;
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
		if (hasFreshFaceLocation) {
			const cv::Rect trackedFaceArea = trackAreaRoi;
			const Point2f rawFaceAnchor = GetDetectionDrivenFaceAnchor(trackedFaceArea);
			Point2f smoothedFaceAnchor = rawFaceAnchor;

			if (m_hasPreviousFaceAnchor) {
				smoothedFaceAnchor = SmoothPoint(
					m_previousFaceAnchor,
					rawFaceAnchor,
					DETECTION_DRIVEN_FACE_SMOOTHING);
				float dx = m_previousFaceAnchor.x - smoothedFaceAnchor.x;
				float dy = m_previousFaceAnchor.y - smoothedFaceAnchor.y;
				ApplyDeadzone(dx, dy, DETECTION_DRIVEN_FACE_DEADZONE_PX);
				xVel = 2.0f * dx;
				yVel = 2.0f * -dy;
			}

			m_previousFaceAnchor = smoothedFaceAnchor;
			m_hasPreviousFaceAnchor = true;
			m_corners.clear();
			m_corners.push_back(smoothedFaceAnchor);

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
					"MediaPipe face-box tracking update: xVel=%.3f yVel=%.3f center=(%d,%d)",
					xVel,
					yVel,
					cvRound(trackedFaceArea.x + trackedFaceArea.width / 2.0f),
					cvRound(trackedFaceArea.y + trackedFaceArea.height / 2.0f));
				s_lastTrackingLog = now;
			}
		}
		else {
			m_corners.clear();
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
