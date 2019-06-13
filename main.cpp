#include "mainwindow.h"
#include <QApplication>
#include<iostream>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("Retinex");
    w.show();

    return a.exec();
}
