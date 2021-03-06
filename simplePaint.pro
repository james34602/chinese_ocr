#-------------------------------------------------
#
# Project created by QtCreator 2017-09-03T14:18:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = simplePaint
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
        canvaswidget.cpp \
        chinese_ocr/matrix.c \
        chinese_ocr/cpthread.c \
        chinese_ocr/misc.c \
        chinese_ocr/chineseocr.c

HEADERS += \
        mainwindow.h \
        canvaswidget.h \
        canvaswidget.cpp \
        chinese_ocr/matrix.h \
        chinese_ocr/cpthread.h \
        chinese_ocr/misc.h \
        chinese_ocr/base64Text.h \
        chinese_ocr/densenet.h \
        chinese_ocr/chineseocr.h

FORMS += \
        mainwindow.ui
