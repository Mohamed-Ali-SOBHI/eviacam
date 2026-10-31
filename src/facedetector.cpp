// SPDX-License-Identifier: GPL-3.0-or-later
#include "facedetector.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace eviacam {
bool FaceDetector::Load(const std::string& cascade, const std::string& model, bool legacy) {
    m_usingYuNet = false;
    m_error.clear();
    try { m_haar.load(cascade); }
    catch (const cv::Exception& e) { m_error = e.what(); }
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 8)
    m_yunet.release();
    if (!legacy) {
        try {
            m_yunet = cv::FaceDetectorYN::create(model, "", cv::Size(320, 320), 0.85f, 0.3f, 5000);
            m_usingYuNet = !m_yunet.empty();
        } catch (const cv::Exception& e) { m_error = e.what(); }
    }
#else
    if (!legacy) m_error = "YuNet requires OpenCV 4.8 or newer; using Haar.";
#endif
    return Ready();
}

bool FaceDetector::Ready() const { return m_usingYuNet || !m_haar.empty(); }

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 8)
std::vector<FaceDetection> FaceDetector::DetectYuNet(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1) cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else bgr = image;
    const double scale = std::min(1.0, 640.0 / std::max(bgr.cols, bgr.rows));
    cv::Mat input;
    cv::resize(bgr, input, cv::Size(), scale, scale, cv::INTER_AREA);
    m_yunet->setInputSize(input.size());
    cv::Mat detected;
    m_yunet->detect(input, detected);
    // A second pass only when the original dark image yielded no face.
    // Limit contrast amplification so sensor noise is not boosted without bound.
    if (detected.empty()) {
        cv::Mat lab;
        cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> planes;
        cv::split(lab, planes);
        if (cv::mean(planes[0])[0] < 85.0) {
            cv::createCLAHE(2.0, cv::Size(8, 8))->apply(planes[0], planes[0]);
            cv::merge(planes, lab);
            cv::cvtColor(lab, input, cv::COLOR_Lab2BGR);
            m_yunet->detect(input, detected);
        }
    }
    std::vector<FaceDetection> result;
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    for (int i = 0; i < detected.rows; ++i) {
        const float* f = detected.ptr<float>(i);
        if (!std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2]) ||
            !std::isfinite(f[3]) || !std::isfinite(f[14])) continue;
        cv::Rect box(cvRound(f[0] / scale), cvRound(f[1] / scale),
                     cvRound(f[2] / scale), cvRound(f[3] / scale));
        box &= bounds;
        if (box.width >= 20 && box.height >= 20 && f[14] >= 0.85f)
            result.push_back({box, f[14]});
    }
    return result;
}
#endif

std::vector<FaceDetection> FaceDetector::Detect(const cv::Mat& image) {
    if (image.empty() || image.cols < 32 || image.rows < 32) return {};
    CV_Assert(image.depth() == CV_8U && (image.channels() == 1 || image.channels() == 3));
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 8)
    if (m_usingYuNet) {
        try { return DetectYuNet(image); }
        catch (const cv::Exception& e) {
            // An unavailable/incompatible network must not take down mouse control.
            m_error = e.what();
            m_usingYuNet = false;
        }
    }
#endif
    if (m_haar.empty()) return {};
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;
    std::vector<cv::Rect> boxes;
    // Exact upstream parameters: this is also the A/B reference implementation.
    m_haar.detectMultiScale(gray, boxes, 1.5, 2,
        cv::CASCADE_FIND_BIGGEST_OBJECT | cv::CASCADE_DO_CANNY_PRUNING, cv::Size(65, 65));
    std::vector<FaceDetection> result;
    for (const auto& box : boxes) result.push_back({box, 1.0f});
    return result;
}

cv::Rect FaceDetector::Select(const std::vector<FaceDetection>& faces, const cv::Rect& previous) {
    if (faces.empty()) return {};
    const FaceDetection* best = &faces.front();
    double bestOverlap = 0;
    if (previous.area() > 0) {
        for (const auto& face : faces) {
            double overlap = static_cast<double>((face.box & previous).area()) /
                (face.box.area() + previous.area() - (face.box & previous).area());
            if (overlap > bestOverlap) { bestOverlap = overlap; best = &face; }
        }
        if (bestOverlap >= 0.2) return best->box;
    }
    return std::max_element(faces.begin(), faces.end(), [](const FaceDetection& a, const FaceDetection& b) {
        return a.box.area() < b.box.area();
    })->box;
}
}
