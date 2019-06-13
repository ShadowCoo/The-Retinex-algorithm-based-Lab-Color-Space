#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->srcImg->clear();
    ui->dstImg->clear();
    QPalette palette;
    palette.setColor(QPalette::Background,QColor(250,250,250));
    ui->srcImg->setAutoFillBackground(true);
    ui->dstImg->setAutoFillBackground(true);
    ui->srcImg->setPalette(palette);
    ui->dstImg->setPalette(palette);
    QObject::connect(ui->ChooseImage,SIGNAL(clicked(bool)),this,SLOT(openFileSlot()));
    QObject::connect(ui->ProcessingImage,SIGNAL(clicked(bool)),this,SLOT(processingImgSlot()));
    QObject::connect(ui->maxScaleSilde,SIGNAL(valueChanged(int)),ui->maxScaleBox,SLOT(setValue(int)));
    QObject::connect(ui->maxScaleBox,SIGNAL(valueChanged(int)),ui->maxScaleSilde,SLOT(setValue(int)));
    QObject::connect(ui->scaleNumSlide,SIGNAL(valueChanged(int)),ui->scaleNumBox,SLOT(setValue(int)));
    QObject::connect(ui->scaleNumBox,SIGNAL(valueChanged(int)),ui->scaleNumSlide,SLOT(setValue(int)));
    QObject::connect(ui->Algorithm,SIGNAL(currentTextChanged(QString)),this,SLOT(algorithmChangeSlot()));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::openFileSlot()
{
    this->filename = QFileDialog::getOpenFileName(this,tr("open image"),".",tr("Image file(*.png *.jpg *.bmp)"));
    if(filename.isEmpty())
    {
        QMessageBox::information(this,"Error Message","Please select a text!");
        return;
    }
    QImage *img = new QImage;
    if(!img->load(filename))
    {
        QMessageBox::information(this,tr("打开图片失败"),tr("打开图片失败"));
        delete img;
        return;
    }
    ui->srcImg->setPixmap(QPixmap::fromImage(*img));
    QImage i = *img;
    transformimg trans;
    Mat srcimg = trans.QImage2Mat(i);
    Mat srcCon = srcimg.clone();
    ui->srcEntropyValue->setText(QString::number(trans.entropy(srcCon), 10, 4));
    ui->srcContrastValue->setText(QString::number(trans.analyse(srcCon), 10, 4));
    ui->srcArticulationValue->setText(QString::number(trans.articulation(srcCon), 10, 4));
    delete img;


    this->algorithmChangeSlot();
}

void MainWindow::processingImgSlot()
{
    if(this->filename == NULL)
    {
        QMessageBox::warning(NULL,"warning","请先选择图片！",QMessageBox::Yes, QMessageBox::Yes);
        return;
    }
    QImage src = ui->srcImg->pixmap()->toImage();
    Msrcr msrcr;
    transformimg trans;
    Mat srcimg = trans.QImage2Mat(src);
    Mat dst;
    vector<double> weights,sigema;


    //qDebug()<<ui->scaleNumSlide->value();
    if(ui->Algorithm->currentText()== "SSR")
    {
        msrcr.Retinex(srcimg,dst,ui->maxScaleSilde->value());
    }
    else if(ui->Algorithm->currentText() == "MSR")
    {
        weights = trans.createWeight(ui->scaleNumSlide->value());
        sigema = trans.createSigema(ui->maxScaleSilde->value(),ui->scaleNumSlide->value());
        msrcr.MultiScaleRetinex(srcimg,dst,weights,sigema);
        weights.clear();
        sigema.clear();
    }
    else
    {
        weights = trans.createWeight(ui->scaleNumSlide->value());
        sigema = trans.createSigema(ui->maxScaleSilde->value(),ui->scaleNumSlide->value());
        msrcr.MultiScaleRetinexCR(srcimg,dst,weights,sigema);
        weights.clear();
        sigema.clear();
    }
    Mat dstCon = dst.clone();
    ui->dstrEntropyValue->setText(QString::number(trans.entropy(dstCon), 10, 4));
    ui->dstContrastValue->setText(QString::number(trans.analyse(dstCon), 10, 4));
    ui->dstArticulationValue->setText(QString::number(trans.articulation(dstCon), 10, 4));
    QImage dstimg = trans.Mat2QImage(dst);
    ui->dstImg->setPixmap(QPixmap::fromImage(dstimg));
    this->reloadImgSlot();
}

void MainWindow::reloadImgSlot()
{
    QImage *img = new QImage;
    if(!img->load(this->filename))
    {
        QMessageBox::information(this,tr("打开图片失败"),tr("打开图片失败"));
        delete img;
        return;
    }
    ui->srcImg->setPixmap(QPixmap::fromImage(*img));
    delete img;
}


void MainWindow::algorithmChangeSlot()
{
    if(ui->Algorithm->currentText() == "SSR")
    {
        ui->scaleNumSlide->setDisabled(true);
        ui->scaleNumBox->setDisabled(true);
    }
    else
    {
        ui->scaleNumSlide->setDisabled(false);
        ui->scaleNumBox->setDisabled(false);
    }
}
