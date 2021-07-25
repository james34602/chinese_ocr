#include "canvaswidget.h"
#define MODEL_NUM (2)
#if MODEL_NUM == 0
#include "simplifiedChineseBest_quantized.c"
#elif MODEL_NUM == 1
#include "traditionalChineseBest_quantized.c"
#else
#include "chineseGeneralweights_quantized.c"
#endif
CanvasWidget::CanvasWidget(QWidget *parent) : QWidget(parent)
{
    //Set up image
    canvasImage = QImage(this->size(), QImage::Format_RGB32);
    //Fill image with wite color as a canvas
    canvasImage.fill(Qt::white);
    //Set drawing flag to false
    drawingActive = false;
    //Set default color to black
    currentColor = Qt::black;
    //Set default brush size to 5
    brushSize = 5;
    init_sample_vector(&imageBuf, 64 * 64 * 3, 100 * 3);
    ocr = (void*)malloc(ChineseOCRRetOCRAllocationSize());
    InitChineseOCR(ocr, (unsigned char*)NNCoeff, 80, 2, 4);
}
#include "chinese_ocr/stb_image_write.h"
void CanvasWidget::onRecognition(unsigned char *image, int w, int h)
{
    //stbi_write_png("img.png", w, h, colourComponent, imageBuf.data, colourComponent * w);
    RecognizeImage(ocr, imageBuf.data, w, h);
    size_t textMemSize;
    wchar_t *ptrText = (wchar_t*)ChineseOCRRetUnicodePointer(ocr, &textMemSize);
    QString txt = QString::fromWCharArray(ptrText);
    emit changeText(txt);
}

CanvasWidget::~CanvasWidget()
{
    clear_sample_vector(&imageBuf);
    FreeChineseOCR(ocr);
    free(ocr);
}

//=========================================================
//Drawing parameters methods
//=========================================================
void CanvasWidget::setColor(QColor selectedColor) {
    currentColor = selectedColor;
}

void CanvasWidget::setBrushSize(int selectedSize) {
    brushSize = selectedSize;
}

void CanvasWidget::clearAll()
{
    canvasImage = QImage(this->size(), QImage::Format_RGB32);
    canvasImage.fill(Qt::white);
    this->update();
    int WIDTH = canvasImage.width();
    int HEIGHT = canvasImage.height();
    push_back_sample_vector(&imageBuf, WIDTH * HEIGHT * colourComponent);
    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < HEIGHT; j++)
        {
            int pos = j * WIDTH * colourComponent + i * colourComponent;
            imageBuf.data[pos] = 255;
            //imageBuf.data[pos] = 255;
            //imageBuf.data[pos + 1] = 255;
            //imageBuf.data[pos + 2] = 255;
        }
    }
    QString txt = QString::fromWCharArray(L"");
    emit changeText(txt);
}

//=========================================================
//Mouse event methods
//=========================================================
void CanvasWidget::mousePressEvent(QMouseEvent *event)
{
    //Check if the button mouse that was clicked was the left one
    if (event->button() == Qt::LeftButton)
    {
        //Save last mouse point
        latestPoint = event->pos();
        //Raise flag to indicate that we are currently drawing something
        drawingActive = true;
    }
}

void CanvasWidget::mouseMoveEvent(QMouseEvent *event)
{
    //Check again if the left mouse button was clicked and whether we are drawing something
    if ((event->buttons() & Qt::LeftButton) && drawingActive)
    {
        //Create a new painter
        QPainter painter(&canvasImage);
        //Set parameters according to settings (hard-coded for now)
        painter.setPen(QPen(currentColor, brushSize, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        //Draw line
        painter.drawLine(latestPoint, event->pos());
        //Save last point
        latestPoint = event->pos();
        //Update canvas
        this->update();
    }
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent *event)
{
    //When the left mouse button is released, set the flag to false
    if (event->button() == Qt::LeftButton)
    {
        drawingActive = false;
        int WIDTH = canvasImage.width();
        int HEIGHT = canvasImage.height();
        push_back_sample_vector(&imageBuf, WIDTH * HEIGHT * colourComponent);
        int pixelRGB;
        unsigned short red, green, blue;
        for (int i = 0; i < WIDTH; i++)
        {
            for (int j = 0; j < HEIGHT; j++)
            {
                pixelRGB = canvasImage.pixel(i, j);
                red = (((uint32_t)((pixelRGB & 0x00ff0000) >> 16) * 6966u) + (1 << (14 - 1))) >> 14;
                green = (((uint32_t)((pixelRGB & 0x0000ff00) >> 8) * 23435u) + (1 << (14 - 1))) >> 14;
                blue = (((uint32_t)(pixelRGB & 0x000000ff) * 2366u) + (1 << (14 - 1))) >> 14;
                int pos = j * WIDTH * colourComponent + i * colourComponent;
                //imageBuf.data[pos] = r;
                //imageBuf.data[pos + 1] = g;
                //imageBuf.data[pos + 2] = b;
                imageBuf.data[pos] = ((red + green + blue) >> 1);
            }
        }
        onRecognition(imageBuf.data, WIDTH, HEIGHT);
    }
}

//=========================================================
//Draw method
//=========================================================
void CanvasWidget::paintEvent(QPaintEvent *event)
{
    QPainter canvasPainter(this);
    canvasPainter.drawImage(this->rect(), canvasImage, canvasImage.rect());
}

//=========================================================
//Resize method
//=========================================================
void CanvasWidget::resizeEvent(QResizeEvent *event)
{
    //On resize delete everything (not an optimal solution to be true...)
    canvasImage = QImage(this->size(), QImage::Format_RGB32);
    canvasImage.fill(Qt::white);
}

//=========================================================
//Various methods
//=========================================================
QImage CanvasWidget::getImage()
{
    return canvasImage;
}
