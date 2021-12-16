// super-stream-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <time.h>

int main()
{
    cv::dnn_superres::DnnSuperResImpl sr;

    /*std::string img_path = "C://CodingProjects//super-stream//inputs//frame0.png";
    cv::Mat img = cv::imread(img_path);*/

    std::string path = "C://CodingProjects//super-stream//models//ESPCN//ESPCN_x2.pb";
    sr.readModel(path);

    sr.setModel("espcn", 2);

    cv::VideoCapture cap;
    if (!cap.open(0))
    {
        return 0;
    }
    std::time_t timer;
    double start = std::time(&timer);
    int frameCount = 0;

    cv::Mat testFrame;
    cap >> testFrame;
    if (testFrame.empty()) return 0;
    int rows = testFrame.rows;
    int cols = testFrame.cols;
    std::cout << "Width: " << cols << " Height: " << rows << std::endl;

    cv::Mat frame;
    cv::Mat smallFrame;
    cv::Mat img_new;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, smallFrame, cv::Size(), 0.5, 0.5);

        sr.upsample(smallFrame, img_new);
        cv::imshow("frame", img_new);
        if (cv::waitKey(10) == 27) break;
        // cv::imwrite("upscaled.png", img_new);
        ++frameCount;
        if (std::time(&timer) - start >= 1)
        {
            std::cout << "FPS: " << frameCount << std::endl;
            frameCount = 0;
            start = std::time(&timer);
        }
    }
    return 0;
}

