//
// Created by aayush on 19/1/19.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;
int main(int argc, char**argv){
    auto IMGFNAME = argv[1];
    auto imgorig = imread(IMGFNAME, IMREAD_GRAYSCALE);
    Mat img;
    GaussianBlur(imgorig, img, Size(7, 7), 0);
    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 41, 10.0);
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", imgorig);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.data[y * img.cols + x] > 0) {
                Mat nimg = Mat::zeros(img.rows, img.cols, CV_8U);
                vector<int> xs;
                vector<int> ys;
                xs.push_back(x);
                ys.push_back(y);
                img.data[y * img.cols + x] = 0;
                nimg.data[y * img.cols + x] = 255;
                int index = 0;
                while (index < xs.size()) {
                    int curx = xs[index], cury = ys[index];
                    index++;
                    for (int p = -1; p < 2; p++) {
                        for (int q = -1; q < 2; q++) {
                            if (img.data[((cury + p) * img.cols) + curx + q] > 0 && curx + q > 0 && curx + q < img.cols && cury + p > 0 && cury + p < img.rows) {
                                img.data[((cury + p) * img.cols) + curx + q] = 0;
                                nimg.data[((cury + p) * img.cols) + curx + q] = 255;
                                xs.push_back(curx + q);
                                ys.push_back(cury + p);
                            }
                        }
                    }
                }
                if (xs.size() > 100) {
                    int vstart, vend = img.rows, hstart, hend = img.cols, vstarted = 0, hstarted = 0;
                    for(int e = 0; e<nimg.rows; e++){
                        if((!vstarted) && sum(nimg.row(e))[0]>0){
                            vstart = e;
                            vstarted = 1;
                        }else if(vstarted && sum(nimg.row(e))[0]==0){
                            vend = e;
                            vstarted = 0;
                        }
                    }
                    for(int w = 0; w<nimg.cols; w++){
                        if((!hstarted) && sum(nimg.col(w))[0]>0){
                            hstart = w;
                            hstarted = 1;
                        }else if(hstarted && sum(nimg.col(w))[0]==0){
                            hend = w;
                            hstarted = 0;
                        }
                    }
                    Mat roi = nimg(Range(vstart, vend), Range(hstart, hend));
                    resize(roi, roi, Size(32, 32), 0, 0, INTER_CUBIC);
                    auto nn = readNetFromTensorflow("./fg.pb");
                    Mat a = Mat::zeros(32, 32, CV_8UC1);
                    nn.setInput(blobFromImage(roi), "ip");
                    auto forward1 = nn.forward("fully_connected_1/Relu");
                    Point minloc, maxloc;
                    minMaxLoc(forward1, NULL, NULL, &minloc, &maxloc);
                    rectangle(imgorig, Point(hstart, vstart), Point(hend, vend), Scalar(0,0, 0));
                    putText(imgorig, boost::lexical_cast<string>(maxloc.x), Point((hstart+hend)/2, (vstart+vend)/2), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 0));
                }
            }
        }
    }
    imshow("Image", imgorig);
    waitKey(0);
    return 0;
}
