#include "transformimg.h"
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <QImage>
using namespace std;
using namespace cv;


//将QImage转换成Mat
Mat transformimg::QImage2Mat(QImage image)
{
    Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}
//将Mat转换为QImage
QImage transformimg::Mat2QImage(const Mat &mat)
{
    if (mat.type() == CV_8UC1)                          // 单通道
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);                       // 灰度级数256
        for (int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        uchar *pSrc = mat.data;                         // 复制mat数据
        for (int row = 0; row < mat.rows; row++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    else if (mat.type() == CV_8UC3)                     // 3通道
    {
        const uchar *pSrc = (const uchar*)mat.data;     // 复制像素
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);    // R, G, B 对应 0,1,2
        return image.rgbSwapped();                      // rgbSwapped是为了显示效果色彩好一些。
    }
    else if (mat.type() == CV_8UC4)                     // 4通道
    {
        const uchar *pSrc = (const uchar*)mat.data;     // 复制像素
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);        // B,G,R,A 对应 0,1,2,3
        return image.copy();
    }
    else
    {
        return QImage();
    }
}

//创建权重
vector<double> transformimg::createWeight(int scale)
{
    vector<double> weight;
    for(int i=0;i<scale;i++)
    {
        weight.push_back(1./scale);
    }
    return weight;
}

vector<double> transformimg::createSigema(int maxScale,int scale)
{
    vector<double> sigema;
    if(scale==2)
    {
        sigema.push_back(maxScale*0.4);
        sigema.push_back(maxScale*0.8);
    }
    else if(scale==3)
    {
        sigema.push_back(maxScale*0.1);
        sigema.push_back(maxScale*0.5);
        sigema.push_back(maxScale);
    }
    else if(scale==4)
    {
        sigema.push_back(maxScale*0.1);
        sigema.push_back(maxScale*0.4);
        sigema.push_back(maxScale*0.7);
        sigema.push_back(maxScale);
    }
    else
    {
        sigema.push_back(maxScale*0.1);
        sigema.push_back(maxScale*0.3);
        sigema.push_back(maxScale*0.5);
        sigema.push_back(maxScale*0.7);
        sigema.push_back(maxScale);
    }
    return sigema;
}

//计算图像方差
double transformimg::analyse(Mat input)
{
    Mat imageGrey;
    cvtColor(input,imageGrey,CV_RGB2GRAY);
    Mat meanValueImage;
    Mat meanStdValueImage;

    meanStdDev(imageGrey,meanValueImage,meanStdValueImage);
    double meanValue = 0.0;
    meanValue = meanStdValueImage.at<double>(0,0);
    return meanValue;
}

//计算图像信息熵
double transformimg::entropy(Mat img)
{
    // 将输入的矩阵为图像
    double temp[256];
    // 清零
    for(int i=0;i<256;i++)
    {
        temp[i] = 0.0;
    }
    // 计算每个像素的累积值
    for(int m=0;m<img.rows;m++)
    {// 有效访问行列的方式
     const uchar* t = img.ptr<uchar>(m);
     for(int n=0;n<img.cols;n++)
     {
        int i = t[n];
        temp[i] = temp[i]+1;
     }
    }

    // 计算每个像素的概率
    for(int i=0;i<256;i++)
    {
        temp[i] = temp[i]/(img.rows*img.cols);
    }

    double result = 0;
    // 根据定义计算图像熵
    for(int i =0;i<256;i++)
    {
        if(temp[i]==0.0)
        {
            result = result;
        }
        else
        {
            result = result-temp[i]*(log(temp[i])/log(2.0));
        }
    }
    return result;
}

//拉普拉斯方法计算图像清晰度
double transformimg::articulation(Mat img)
{
    Mat imageGrey;
    cvtColor(img,imageGrey,CV_RGB2GRAY);
    Mat imageLap;
    Laplacian(imageGrey,imageLap,CV_16U);

    double meanValue = 0.0;
    meanValue = mean(imageLap)[0];
    return meanValue;
}
