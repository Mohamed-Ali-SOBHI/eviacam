// SPDX-License-Identifier: GPL-3.0-or-later
#include "visionpipeline.h"
#include "paths.h"
#include <wx/init.h>
#include <wx/utils.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>

static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char** argv) {
    wxInitializer wx;
    if (!wx.IsOk()) return 2;
    cv::setNumThreads(2);
    eviacam::SetDataDir(wxString::FromUTF8(argc > 1 ? argv[1] : "build"));
    try {
        CVisionPipeline pipeline;
        pipeline.SetCpuUsage(CVisionPipeline::CPU_HIGH);
        pipeline.SetTimeout(1);
        pipeline.SetEnableWhenFaceDetected(true);
        require(pipeline.IsTrackFaceAllowed(), "detector unavailable");
        require(!pipeline.IsFaceDetected(), "face reported before first detection");
        auto original=cv::imread("tests/generated/astronaut_w32_b0.3.png");
        require(!original.empty(), "missing fixture");
        bool acquired=false;
        for(int i=0;i<120;++i) {
            auto frame=original.clone();
            float x,y;
            pipeline.ProcessImage(frame,x,y);
            require(std::isfinite(x) && std::isfinite(y), "nonfinite tracking motion");
            acquired=acquired || pipeline.IsFaceDetected();
            wxMilliSleep(15);
        }
        require(acquired, "small dark face was never acquired asynchronously");
        cv::Rect area;
        pipeline.GetTrackAreaControl()->GetBoxImg(original,area);
        require((area & cv::Rect(135,90,55,60)).area()>0, "tracking area not on face");
        // A known translation must reach the tracker as motion, not just a rectangle.
        double accumulatedX=0;
        for(int i=1;i<=10;++i) {
            cv::Mat frame;
            cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,i,0,1,0);
            cv::warpAffine(original,frame,transform,original.size());
            float x,y;
            require(pipeline.ProcessImage(frame,x,y), "valid translated frame rejected");
            accumulatedX+=x;
            wxMilliSleep(15);
        }
        require(accumulatedX < -8 && accumulatedX > -40, "tracker did not follow rightward translation");
        bool allowed=true;
        for(int i=0;i<120;++i) {
            cv::Mat frame(240,320,CV_8UC3,cv::Scalar(20,20,20));
            float x,y;
            allowed=pipeline.ProcessImage(frame,x,y);
            wxMilliSleep(15);
        }
        require(!allowed && !pipeline.IsFaceDetected(), "lost face did not stop motion gate");
        for(int size : {64,240,480,120}) {
            cv::Mat frame(size,size,CV_8UC3,cv::Scalar(0,0,0));
            float x,y;
            pipeline.ProcessImage(frame,x,y);
            require(std::isfinite(x) && std::isfinite(y), "resolution change invalid motion");
        }
        cv::Mat empty;
        float x=99,y=99;
        require(!pipeline.ProcessImage(empty,x,y) && x==0 && y==0, "empty frame unsafe");
        pipeline.SetCpuUsage(CVisionPipeline::CPU_LOW);
        require(pipeline.GetCpuUsage()==CVisionPipeline::CPU_LOW, "low CPU setting broken");
        std::cout << "PASS: async small/dark face acquisition, tracker bounds, timeout, resize, empty input, CPU setting, thread shutdown\n";
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
    return 0;
}
