#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;
using namespace cv;
using namespace boost::filesystem;

const uchar *getPixel(Mat *z, int x, int y) {
    cout << z->data << endl;
    return z->data + (y * z->cols) + x;
}

int main() {
    create_directories("./numbers");
    Mat img = imread("./mnist2.png", IMREAD_GRAYSCALE);
    //resize(img, img, Size(0, 0), 0.9, 0.9);
    GaussianBlur(img, img, Size(7, 7), 0);
    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 41, 10.0);
    namedWindow("adpthr", WINDOW_NORMAL);
    imshow("adpthr", img);
    waitKey(100000);
    cout << img.type() << endl << CV_8UC1 << endl;
    cout << img.data[100 * img.cols + 100];
    imshow("adpthr", img);
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
                    cout<<endl<<vstart<<endl<<vend<<endl<<hstart<<endl<<hend<<endl<<endl;
                    Mat roi = nimg(Range(vstart, vend), Range(hstart, hend));
                    resize(roi, roi, Size(32, 32), 0, 0, INTER_CUBIC);
                    path p("./numbers");
                    int file_count = 0;
                    directory_iterator di = directory_iterator(p);
                    while (di != directory_iterator()) {
                        cout << di->path() << endl;
                        file_count++;
                        di++;
                    }
                    string n = boost::lexical_cast<string>(file_count);
                    string s = "./numbers/Img" + n + ".png";
                    cout << s;
                    imshow("number", roi);
                    waitKey(0);
                    //imwrite(s, roi);
                }
            }
        }
    }
    waitKey(10000);

}