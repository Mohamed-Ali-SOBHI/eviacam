// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

namespace eviacam {
struct FaceDetection {
    cv::Rect box;
    float confidence;
};

// One instance per worker: OpenCV networks/classifiers are not shared across threads.
class FaceDetector {
public:
    bool Load(const std::string& cascade, const std::string& model, bool legacy = false);
    std::vector<FaceDetection> Detect(const cv::Mat& image);
    bool Ready() const;
    bool UsingYuNet() const { return m_usingYuNet; }
    const std::string& LastError() const { return m_error; }
    static cv::Rect Select(const std::vector<FaceDetection>& faces, const cv::Rect& previous);
private:
    cv::CascadeClassifier m_haar;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 8)
    cv::Ptr<cv::FaceDetectorYN> m_yunet;
    std::vector<FaceDetection> DetectYuNet(const cv::Mat& image);
#endif
    bool m_usingYuNet = false;
    std::string m_error;
};
}
