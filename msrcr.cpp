#include "MSRCR.h"
#include<opencv2\opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include <math.h>

using namespace cv;
using namespace std;

/*===========================================================
* 函数: CreateKernel
* 说明：创建一个标准化的一维高斯核；
* 参数：
*   double sigma: 高斯核标准偏差
* 返回值：
*   double*: 长度为((6*sigma)/2) * 2 + 1的double数组
=============================================================*/
vector<double> Msrcr::CreateKernel(double sigma)
{
    int i, x, filter_size;
    vector<double> filter;
    double sum;
    //为sigma设定上限
    if (sigma > 300) sigma = 300;

    //获取需要的滤波尺寸，且强制为奇数
    filter_size = (int)floor(sigma * 2) / 2;
    filter_size = filter_size * 2 + 1;
    //计算指数
    sum = 0;
    for (i = 0; i < filter_size; i++)
    {
        double tmpValue;
        x = i - (filter_size / 2);
        tmpValue = exp(-(x*x) / (2 * sigma*sigma));
        filter.push_back(tmpValue);
        sum += tmpValue;
    }

    //归一化计算
    for (i = 0; i < filter_size; i++)
        filter[i] /= sum;

    return filter;

}

/*===========================================================
* 函数: CreateFastKernel
* 说明：创建一个近似浮点的整数类型（左移8bits）的快速高斯核；
* 参数：
*   double sigma: 高斯核标准偏差
* 返回值：
*   double*: 长度为((6*sigma)/2) * 2 + 1的int数组
=============================================================*/
vector<int> Msrcr::CreateFastKernel(double sigma)
{
    vector<double> fp_kernel;
    vector<int> kernel;
    int i, filter_size;
    //设置上限
    if (sigma > 300) sigma = 300;
    //获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma * 6) / 2;
    filter_size = filter_size * 2 + 1;
    //创建内核
    fp_kernel = CreateKernel(sigma);
    //double内核转为int型
    for (i = 0; i < filter_size; i++)
    {
        int tmpValue;
        tmpValue = double2int(fp_kernel[i]);
        kernel.push_back(tmpValue);
    }
    return kernel;
}

/*===========================================================
* 函数：FilterGaussian
* 说明：通过内核计算高斯卷积，内核由sigma值得到，且在内核两端值相等；
* 参数：
*   img: 被滤波的IplImage*类型图像
*   double sigma: 高斯核标准偏差
=============================================================
*/
void Msrcr::FilterGaussian(IplImage* img, double sigma)
{
    int i, j, k, source, filter_size;
    vector<int> kernel;
    IplImage* temp;
    int v1, v2, v3;

    //设置上限
    if (sigma > 300) sigma = 300;

    //获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma * 6) / 2;
    filter_size = filter_size * 2 + 1;

    //创建内核
    kernel = CreateFastKernel(sigma);

    temp = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);

    //X轴滤波
    for (j = 0; j < temp->height; j++)
    {
        for (i = 0; i < temp->width; i++)
        {
            //内层循环已经展开
            v1 = v2 = v3 = 0;
            for (k = 0; k < filter_size; k++)
            {
                source = i + filter_size / 2 - k;

                if (source < 0) source *= -1;
                if (source > img->width - 1) source = 2 * (img->width - 1) - source;

                v1 += kernel[k] * (unsigned char)pc(img, source, j, 0);
            }

            //设置像素点的值
            pc(temp, i, j, 0) = (char)int2smallint(v1);
        }
    }

    //Y轴滤波
    for (j = 0; j < img->height; j++)
    {
        for (i = 0; i < img->width; i++)
        {
            v1 = v2 = v3 = 0;
            for (k = 0; k < filter_size; k++)
            {
                source = j + filter_size / 2 - k;

                if (source < 0) source *= -1;
                if (source > temp->height - 1) source = 2 * (temp->height - 1) - source;

                v1 += kernel[k] * (unsigned char)pc(temp, i, source, 0);
            }
            pc(img, i, j, 0) = (char)int2smallint(v1);
        }
    }

    cvReleaseImage(&temp);
}

/*===========================================================
* 函数：FilterGaussian
* 说明：通过内核计算高斯卷积，内核由sigma值得到，且在内核两端值相等；
* 参数：
*   Mat src: 输入图像
*   Mat &dst: 输出图像
*   double sigma: 高斯核标准偏差
=============================================================
*/
void Msrcr::FilterGaussian(Mat src, Mat &dst, double sigma)
{
    IplImage tmp_ipl;
    tmp_ipl = IplImage(src);
    FilterGaussian(&tmp_ipl, sigma);
    dst = cvarrToMat(&tmp_ipl);
}

/*===========================================================
* 函数：FastFilter
* 说明：给出任意大小的sigma值，都可以通过使用图像金字塔与可分离滤波器计算高斯卷积；
* 参数：
*   IplImage *img: 被滤波的图像
*   double sigma: 高斯核标准偏差
=============================================================
*/
void Msrcr::FastFilter(IplImage *img, double sigma)
{
    int filter_size;

    //设置上限
    if (sigma > 300) sigma = 300;

    //获取需要的滤波尺寸，且强制为奇数；
    filter_size = (int)floor(sigma * 6) / 2;
    filter_size = filter_size * 2 + 1;
    //如果3 * sigma小于一个像素，则直接退出
    if (filter_size < 3) return;

    //处理方式：(1) 滤波  (2) 高斯光滑处理  (3) 递归处理滤波器大小
    if (filter_size < 10) {

#ifdef USE_EXACT_SIGMA
        FilterGaussian(img, sigma);
#else
        cvSmooth(img, img, CV_GAUSSIAN, filter_size, filter_size);
#endif

    }
    else
    {
        if (img->width < 2 || img->height < 2) return;
        IplImage* sub_img = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);
        cvPyrDown(img, sub_img);
        FastFilter(sub_img, sigma / 2.0);
        cvResize(sub_img, img, CV_INTER_LINEAR);
        cvReleaseImage(&sub_img);
    }
}

/*===========================================================
* 函数：FastFilter
* 说明：给出任意大小的sigma值，都可以通过使用图像金字塔与可分离滤波器计算高斯卷积；
* 参数：
*   Mat src: 输入图像
*   Mat &dst: 输出图像
*   double sigma: 高斯核标准偏差
=============================================================
*/
void Msrcr::FastFilter(Mat src, Mat &dst, double sigma)
{
    IplImage tmp_ipl;
    tmp_ipl = IplImage(src);
    FastFilter(&tmp_ipl, sigma);
    dst = cvarrToMat(&tmp_ipl);
}

/*===========================================================
* 函数：Retinex
* 说明：单通道SSR方法，基础Retinex复原算法。原图像和被滤波的图像需要被转换到
*   对数域，并做减运算；
* 参数：
*   IplImage *img: 被滤波的图像
*   double sigma: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
=============================================================
*/
void Msrcr::Retinex(IplImage *img, double sigma, int gain, int offset)
{
    IplImage *A, *fA, *fB, *fC;


    //初始化缓存图像
    fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);


    //计算对数图像
    cvConvert(img, fA);
    cvLog(fA, fB);


    //计算滤波后模糊图像的对数图像
    A = cvCloneImage(img);
    FastFilter(A, sigma);
    cvConvert(A, fA);
    cvLog(fA, fC);


    //计算两图像之差
    cvSub(fB, fC, fA);

    //重建图像
    cvConvertScale(fA, img, gain, offset);

    //释放缓存图像
    cvReleaseImage(&A);
    cvReleaseImage(&fA);
    cvReleaseImage(&fB);
    cvReleaseImage(&fC);

}

/*===========================================================
* 函数：Retinex
* 说明：单通道SSR方法，基础Retinex复原算法。原图像和被滤波的图像需要被转换到
*   对数域，并做减运算；
* 参数：
*   Mat src: 输入图像
*   Mat &dst: 输出图像
*   double sigma: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
=============================================================
*/
void Msrcr::Retinex(Mat src, Mat &dst, double sigma, int gain, int offset)
{
    IplImage tmp_ipl;
    tmp_ipl = IplImage(src);
    Retinex(&tmp_ipl, sigma, gain, offset);
    dst = cvarrToMat(&tmp_ipl);
    vector<Mat> channels, channels2;
    split(src, channels);
    Mat src2 = channels[1];
    Mat src3 = channels[2];
    split(dst, channels2);
    Mat channel = channels2[0];
    channels.clear();
    channels.push_back(channel);
    channels.push_back(src2);
    channels.push_back(src3);
    merge(channels, dst);

}

/*===========================================================
* 函数：MultiScaleRetinex
* 说明：多通道MSR算法。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。
* 通常情况下，三个权重范围选择低、中、高标准偏差；
*
* 参数：
*   IplImage *img: 被滤波的图像
*   vector<double> weights: 通道权重
*   vector<double> sigmas: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
=============================================================
*/
void Msrcr::MultiScaleRetinex(IplImage *img, vector<double> weights, vector<double> sigmas, int gain, int offset)
{
    int i;
    double weight;
    int scales = sigmas.size();
    IplImage *A, *fA, *fB, *fC;

    //初始化缓存图像
    fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);


    //计算的对数图像
    cvConvert(img, fA);
    cvLog(fA, fB);

    //根据权重归一化
    for (i = 0, weight = 0; i < scales; i++)
        weight += weights[i];

    if (weight != 1.0) cvScale(fB, fB, weight);

    //各尺度上进行滤波操作
    for (i = 0; i < scales; i++)
    {
        A = cvCloneImage(img);
        double tmp = sigmas[i];
        FastFilter(A, tmp);

        cvConvert(A, fA);
        cvLog(fA, fC);
        cvReleaseImage(&A);

        //计算权重后两图像之差
        cvScale(fC, fC, weights[i]);
        cvSub(fB, fC, fB);
    }

    //重建图像
    cvConvertScale(fB, img, gain, offset);

    //释放内存
    cvReleaseImage(&fA);
    cvReleaseImage(&fB);
    cvReleaseImage(&fC);
}

/*===========================================================
* 函数：MultiScaleRetinex
* 说明：多通道MSR算法。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。
* 通常情况下，三个权重范围选择低、中、高标准偏差；
*
* 参数：
*   Mat src: 输入图像
*   Mat &dst: 输出图像
*   vector<double> weights: 通道权重
*   vector<double> sigmas: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
=============================================================
*/
void Msrcr::MultiScaleRetinex(Mat src, Mat &dst, vector<double> weights, vector<double> sigmas, int gain, int offset)
{
    IplImage tmp_ipl;

    tmp_ipl = IplImage(src);
    MultiScaleRetinex(&tmp_ipl, weights, sigmas, gain, offset);
    dst = cvarrToMat(&tmp_ipl);

    vector<Mat> channels, channels2;
    split(src, channels);
    Mat src2 = channels[1];
    Mat src3 = channels[2];
    split(dst, channels2);
    Mat channel = channels2[0];
    channels.clear();
    channels.push_back(channel);
    channels.push_back(src2);
    channels.push_back(src3);
    merge(channels, dst);
}

/*===========================================================
* 函数：MultiScaleRetinexCR
* 说明：MSRCR算法，MSR算法加上颜色修复。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。
* 通常情况下，三个权重范围选择低、中、高标准偏差；之后，颜色修复权重应用于每个颜色通道中；
*
* 参数：
*   IplImage *img: 被滤波的图像
*   double sigma: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
*   double restoration_factor: 控制颜色修复的非线性
*   double color_gain: 控制颜色修复增益
=============================================================
*/
void Msrcr::MultiScaleRetinexCR(IplImage *img, vector<double> weights, vector<double> sigmas,
                                int gain, int offset, double restoration_factor, double color_gain)
{
    int i;
    double weight;
    //尺度数
    int scales = sigmas.size();
    IplImage *A, *B, *C, *fA, *fB, *fC, *fsA, *fsB, *fsC, *fsD, *fsE, *fsF;
    //初始化缓存图像
    fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
    fsA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
    fsB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
    fsC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
    fsD = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
    fsE = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
    fsF = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);

    //计算对数图像
    cvConvert(img, fB);
    cvLog(fB, fA);

    //根据权重归一化
    for (i = 0, weight = 0; i < scales; i++)
        weight += weights[i];

    if (weight != 1.0) cvScale(fA, fA, weight);

    //各尺度上进行滤波操作
    for (i = 0; i < scales; i++) {
        A = cvCloneImage(img);
        FastFilter(A, sigmas[i]);

        cvConvert(A, fB);
        cvLog(fB, fC);
        cvReleaseImage(&A);

        //计算权重后两图像之差
        cvScale(fC, fC, weights[i]);
        cvSub(fA, fC, fA);
    }


    //颜色修复
    if (img->nChannels > 1) {
        A = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);
        B = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);
        C = cvCreateImage(cvSize(img->width, img->height), img->depth, 1);

        //将图像分割为若干通道，类型转换为浮点型，并存储通道数据之和
        cvSplit(img, A, B, C, NULL);
        cvConvert(A, fsA);
        cvConvert(B, fsB);
        cvConvert(C, fsC);

        cvReleaseImage(&A);
        cvReleaseImage(&B);
        cvReleaseImage(&C);

        //求和
        cvAdd(fsA, fsB, fsD);
        cvAdd(fsD, fsC, fsD);

        //带权重矩阵归一化
        cvDiv(fsA, fsD, fsA, restoration_factor);
        cvDiv(fsB, fsD, fsB, restoration_factor);
        cvDiv(fsC, fsD, fsC, restoration_factor);

        cvConvertScale(fsA, fsA, 1, 1);
        cvConvertScale(fsB, fsB, 1, 1);
        cvConvertScale(fsC, fsC, 1, 1);

        // 带权重矩阵求对数
        cvLog(fsA, fsA);
        cvLog(fsB, fsB);
        cvLog(fsC, fsC);

        //将Retinex图像切分为三个数组，按照权重和颜色增益重新组合
        cvSplit(fA, fsD, fsE, fsF, NULL);

        cvMul(fsD, fsA, fsD, color_gain);
        cvMul(fsE, fsB, fsE, color_gain);
        cvMul(fsF, fsC, fsF, color_gain);

        cvMerge(fsD, fsE, fsF, NULL, fA);
    }

    //恢复图像
    cvConvertScale(fA, img, gain, offset);

    //释放缓存图像
    cvReleaseImage(&fA);
    cvReleaseImage(&fB);
    cvReleaseImage(&fC);
    cvReleaseImage(&fsA);
    cvReleaseImage(&fsB);
    cvReleaseImage(&fsC);
    cvReleaseImage(&fsD);
    cvReleaseImage(&fsE);
    cvReleaseImage(&fsF);
}

/*===========================================================
* 函数：MultiScaleRetinexCR
* 说明：MSRCR算法，MSR算法加上颜色修复。原图像和一系列被滤波的图像转换到对数域，并与带权重的原图像做减运算。
* 通常情况下，三个权重范围选择低、中、高标准偏差；之后，颜色修复权重应用于每个颜色通道中；
*
* 参数：
*   Mat src: 输入图像
*   Mat &dst: 输出图像
*   double sigma: 高斯核标准偏差
*   int gain: 图像像素值改变范围的增益
*   int offset: 图像像素值改变范围的偏移量
*   double restoration_factor: 控制颜色修复的非线性
*   double color_gain: 控制颜色修复增益
=============================================================
*/
void Msrcr::MultiScaleRetinexCR(Mat src, Mat &dst, vector<double> weights, vector<double> sigmas,
                                int gain, int offset, double restoration_factor, double color_gain)
{
    IplImage tmp_ipl;
    tmp_ipl = IplImage(src);
    MultiScaleRetinexCR(&tmp_ipl, weights, sigmas, gain, offset, restoration_factor, color_gain);
    dst = cvarrToMat(&tmp_ipl);

    vector<Mat> channels, channels2;
    split(src, channels);
    Mat src2 = channels[1];
    Mat src3 = channels[2];
    split(dst, channels2);
    Mat channel = channels2[0];
    channels.clear();
    channels.push_back(channel);
    channels.push_back(src2);
    channels.push_back(src3);
    merge(channels, dst);
}
