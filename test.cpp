#include "cv.h"
#include "highgui.h"
#include "MSRCR.h"
#include<opencv2\opencv.hpp>
#include<opencv\highgui.h>

using namespace cv;
using namespace std;
vector<double> createScale(int scale, vector<double> weight);
int main()
{
    vector<double> sigema, sigema1;
    vector<double> weight;
    int scale = 2;
    /*for (int i = 0; i < 3; i++)
        weight.push_back(1. / 3);*/
    weight = createScale(scale,weight);
    // 由于MSRCR.h中定义了宏USE_EXTRA_SIGMA，所以此处的vector<double> sigema并没有什么意义
    sigema.push_back(30);
    sigema.push_back(150);
    sigema.push_back(300);
    sigema1.push_back(20);
    sigema1.push_back(60);
    sigema1.push_back(100);
    char key;
    Mat img, imgdst, imgg, imggdst;
    Msrcr msrcr;
    img = imread("16.jpg");
    imshow("Frame", img);
    cv::cvtColor(img, imgg, CV_RGB2Lab);
    //msrcr.Retinex(imgg, imgdst,50, 128, 128);
    //msrcr.MultiScaleRetinex(imgg, imgdst, weight, sigema, 128, 128);
    msrcr.MultiScaleRetinexCR(imgg, imgdst, weight, sigema1, 128, 128);
    cv::cvtColor(imgdst, imggdst, CV_Lab2RGB);
    imshow("dst", imggdst);
    key = (char)cvWaitKey(0);
    return 0;
}


vector<double> createScale(int scale, vector<double> weight) {
    for (int i = 0; i < scale;i++) {
        weight.push_back(1. / scale);
    }
    return weight;
}
