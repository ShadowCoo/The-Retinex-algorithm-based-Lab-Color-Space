#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QString>
#include <QTextStream>
#include <opencv/cv.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <math.h>
#include "msrcr.h"
#include "transformimg.h"

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    QString filename;
public:

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void openFileSlot();
    void processingImgSlot();
    void reloadImgSlot();
    void algorithmChangeSlot();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
