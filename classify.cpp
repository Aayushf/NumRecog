//
// Created by aayush on 18/1/19.
//
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
int main(){
    auto diritr = boost::filesystem::directory_iterator("./numbers");
    boost::filesystem::directory_iterator end;
    while(diritr!=end){
        std::string name = diritr->path().string();
        std::cout<<name<<std::endl;
        auto mat = cv::imread(name, cv::IMREAD_GRAYSCALE);
        cv::imshow("img", mat);
        auto key = cv::waitKey(0)&0xFF;
        std::cout<<(char)key<<std::endl;
        auto fname = "__"+diritr->path().filename().string();
        fname[0] = (char)key;
        auto fpath = "./labelled/"+fname;
        std::cout<<fpath<<std::endl;
        cv::imwrite(fpath, mat);
        boost::filesystem::remove(name);
        diritr++;
    }










    return 0;
}
