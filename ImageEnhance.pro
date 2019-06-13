#-------------------------------------------------
#
# Project created by QtCreator 2019-02-09T08:57:51
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageEnhance
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    msrcr.cpp \
    transformimg.cpp

HEADERS += \
        mainwindow.h \
    msrcr.h \
    transformimg.h

FORMS += \
        mainwindow.ui

INCLUDEPATH+= D:\OpenCV-MinGW-Build\include\opencv  \
                D:\OpenCV-MinGW-Build\include\opencv2\
                D:\OpenCV-MinGW-Build\include

LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_core341.dll
LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_highgui341.dll
LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_imgcodecs341.dll
LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_imgproc341.dll
LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_features2d341.dll
LIBS +=  D:\OpenCV-MinGW-Build\bin\libopencv_calib3d341.dll
