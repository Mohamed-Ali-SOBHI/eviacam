// SPDX-License-Identifier: GPL-3.0-or-later
// No VideoCapture/device access. Exercises exactly the detector linked into the app.
#include "facedetector.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "Usage: face_benchmark root manifest.tsv results.csv\n"; return 2; }
    try {
        cv::setNumThreads(2);
        const std::string root = argv[1];
        eviacam::FaceDetector detectors[2];
        for (int i=0; i<2; ++i)
            if (!detectors[i].Load(root+"/src/data/haarcascade_frontalface_default.xml",
                                  root+"/models/face_detection_yunet_2023mar.onnx", i==0)) return 3;
        if (!detectors[1].UsingYuNet()) return 4;
        std::ifstream manifest(argv[2]);
        std::ofstream output(argv[3]);
        if (!manifest || !output) return 5;
        output << "case,group,detector,expected,detected,true_positive,false_positive,best_iou,ms\n";
        std::string line;
        int cases=0;
        // Exclude one-time OpenCV/DNN initialization from per-frame timings.
        cv::Mat warmup(240,320,CV_8UC3,cv::Scalar(80,80,80));
        for (auto& d : detectors) for (int n=0; n<3; ++n) d.Detect(warmup);
        while (std::getline(manifest,line)) {
            std::istringstream row(line);
            std::string id,group,path;
            int x,y,w,h;
            if (!(row >> id >> group >> path >> x >> y >> w >> h)) return 6;
            const auto image = cv::imread(root+"/"+path);
            if (image.empty()) return 7;
            const cv::Rect truth(x,y,w,h);
            for (int i=0; i<2; ++i) {
                const auto started=std::chrono::steady_clock::now();
                auto faces=detectors[i].Detect(image);
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                double best=0;
                for (const auto& f:faces) {
                    const int intersection=(f.box & truth).area();
                    const int unionArea=f.box.area()+truth.area()-intersection;
                    if (unionArea) best=std::max(best, double(intersection)/unionArea);
                }
                const int tp=truth.area()>0 && best>=0.3;
                output << id << ',' << group << ',' << (i?"yunet":"haar") << ','
                       << (truth.area()>0) << ',' << faces.size() << ',' << tp << ','
                       << faces.size()-tp << ',' << best << ',' << ms << '\n';
                if (i && !detectors[i].UsingYuNet()) return 8;
            }
            ++cases;
        }
        if (!cases) return 9;
        std::cout << "Completed " << cases << " cases, two detectors. OpenCV " << CV_VERSION << "\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 10; }
}
