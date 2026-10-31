// SPDX-License-Identifier: GPL-3.0-or-later
#include "facedetector.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        cv::setNumThreads(2);
        const std::string root = argc > 1 ? argv[1] : ".";
        const auto cascade = root + "/src/data/haarcascade_frontalface_default.xml";
        const auto model = root + "/models/face_detection_yunet_2023mar.onnx";
        eviacam::FaceDetector detector, legacy, fallback, unavailable;
        require(detector.Load(cascade, model), "detector not ready");
        require(detector.UsingYuNet(), "test silently fell back to Haar");
        require(legacy.Load(cascade, model, true) && !legacy.UsingYuNet(), "legacy mode ignored");
        require(fallback.Load(cascade, "nonexistent.onnx") && !fallback.UsingYuNet(), "fallback broken");
        require(!unavailable.Load("nonexistent.xml", "nonexistent.onnx"), "missing models accepted");
        require(detector.Detect(cv::Mat()).empty(), "empty input not handled");
        for (const auto& size : {cv::Size(320,240), cv::Size(640,480), cv::Size(7,9)}) {
            require(detector.Detect(cv::Mat(size, CV_8UC3, cv::Scalar(0,0,0))).empty(), "black frame false positive");
        }
        const auto image = cv::imread(root + "/tests/data/astronaut.png");
        require(!image.empty(), "missing astronaut fixture: run prepare_benchmark.py");
        const auto faces = detector.Detect(image);
        require(detector.UsingYuNet() && !faces.empty(), "real face not detected by YuNet");
        for (const auto& f : faces) {
            require(f.box.area() > 0 && (f.box & cv::Rect(0,0,image.cols,image.rows)) == f.box,
                    "out of bounds detection");
        }
        // This coarse upstream Haar configuration misses the astronaut portrait.
        // Verify fallback on a known Haar-positive fixture and compare its exact output.
        const auto haarPositive = cv::imread(root + "/tests/data/lena.jpg");
        require(!haarPositive.empty(), "missing Haar-positive fixture");
        const auto reference = legacy.Detect(haarPositive);
        const auto recovered = fallback.Detect(haarPositive);
        require(!reference.empty() && recovered.size() == reference.size(), "fallback differs from Haar");
        for (size_t i=0; i<reference.size(); ++i)
            require(recovered[i].box == reference[i].box, "fallback rectangle differs from Haar");
        const std::vector<eviacam::FaceDetection> two = {{{10,10,40,40},0.95f}, {{140,30,90,90},0.99f}};
        require(eviacam::FaceDetector::Select(two, {}) == two[1].box, "initial largest-face selection");
        require(eviacam::FaceDetector::Select(two, {12,12,40,40}) == two[0].box, "target continuity");
        require(eviacam::FaceDetector::Select({}, {}).empty(), "empty selection");
        std::cout << "PASS: model, legacy, fallback, empty/black/tiny input, bounds, real face, target continuity\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
