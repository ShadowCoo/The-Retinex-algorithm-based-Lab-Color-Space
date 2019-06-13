#ifndef TRANSFORMIMG_H
#define TRANSFORMIMG_H

#include <opencv/cv.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <QImage>


using namespace std;
using namespace cv;

class transformimg{
public:
    Mat QImage2Mat(QImage image);
    QImage Mat2QImage(const Mat& mat);
    vector<double> createWeight(int scale);
    vector<double> createSigema(int maxScale,int scale);
    double analyse(Mat input);
    double entropy(Mat img);
    double articulation(Mat img);
};

#endif // TRANSFORMIMG_H
