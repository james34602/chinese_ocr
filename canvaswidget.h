#ifndef CANVASWIDGET_H
#define CANVASWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
extern "C"
{
#include "chinese_ocr/densenet.h"
#include "chinese_ocr/chineseocr.h"
}
class CanvasWidget : public QWidget
{
    typedef struct
    {
        unsigned long long inuse, capacity, grow_num;
        unsigned char *data;
    } sample_vector;
    void init_sample_vector(sample_vector *s, int init_capacity, int grow)
    {
        s->data = (unsigned char*)malloc(init_capacity * sizeof(unsigned char));
        for (int i = 0; i < init_capacity; i++)
            s->data[i] = 255u;
        s->inuse = 0;
        s->capacity = init_capacity;
        s->grow_num = grow;
    }
    void push_back_sample_vector(sample_vector *s, int lenS)
    {
        if ((lenS + (s->grow_num >> 1)) > s->capacity)
        {
            s->capacity += (s->grow_num + lenS);
            s->data = (unsigned char*)realloc(s->data, s->capacity * sizeof(unsigned char));
            unsigned char *ptr = s->data + s->inuse;
            for (int i = 0; i < s->capacity - s->inuse; i++)
                ptr[i] = 255u;
            s->inuse += lenS;
        }
    }
    void clear_sample_vector(sample_vector *s)
    {
        s->inuse = 0;
    }
    void free_sample_vector(sample_vector *s)
    {
        free(s->data);
    }
    Q_OBJECT
public:
    explicit CanvasWidget(QWidget *parent = nullptr);
    ~CanvasWidget();

    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void paintEvent(QPaintEvent *event);
    virtual void resizeEvent(QResizeEvent *event);

    void setColor(QColor selectedColor);
    void setBrushSize(int selectedSize);
    void clearAll();

    QImage getImage();
signals:
    void changeText(const QString &txt);
private:
    //Canvas
    QImage canvasImage;
    //Flag to indicate whether we are currently drawing something (used on mouse events)
    bool drawingActive;
    //Latest mouse point
    QPoint latestPoint;
    //Color variable
    QColor currentColor;
    //Brush size variable
    int brushSize;
    sample_vector imageBuf;
    void onRecognition(unsigned char *image, int w, int h);
    const int colourComponent = 1;
    void *ocr;
};

#endif // CANVASWIDGET_H
