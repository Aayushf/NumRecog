//
// Created by aayush on 19/1/19.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace std;
int main(){
    auto nn = readNetFromTensorflow("./fg.pb");
    auto blb = imread("./labelled/2_Img299.png", IMREAD_GRAYSCALE);
    Mat a = Mat::zeros(32, 32, CV_8UC1);
    nn.setInput(blobFromImage(blb), "ip");
    auto forward1 = nn.forward("fully_connected_1/Relu");
    cout << forward1;
    Point minloc, maxloc;
    minMaxLoc(forward1, NULL, NULL, &minloc, &maxloc);
    cout<<maxloc.x;
    return 0;
}
