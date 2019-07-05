//
// Created by aayush on 18/1/19.
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
int main(){
    auto m = cv::imread("./numbers/Img0.png", cv::IMREAD_GRAYSCALE);
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", m);
    cv::waitKey(1000000);
    return 0;
}
