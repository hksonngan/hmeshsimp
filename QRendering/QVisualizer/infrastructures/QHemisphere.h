/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QHemisphere.h
 * @brief   QHemisphere class definition.
 * 
 * This file defines a widget which can be used to show the visibilities of different views.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/26
 */

#ifndef QHEMISPHERE_H
#define QHEMISPHERE_H

#include <QWidget>

class QHemisphere : public QWidget
{
    Q_OBJECT

public:
    QHemisphere(QWidget *parent);
    ~QHemisphere();

public slots:
    void slotInitViewEntropy(::size_t viewSize, float *viewEntropy, float minEntropy, float maxEntropy);
    void slotUpdateViewEntropy(double entropy);
    void slotMarkePoint(int type, int offset);

protected:
    void paintEvent(QPaintEvent *event);
    void drawSquare(QPainter &painter, const QColor& pen, const QColor& brush, const qreal &x, const qreal &y, const qreal &size);
    void drawCircle(QPainter &painter, const QColor& pen, const QColor& brush, const qreal &x, const qreal &y, const qreal &radius);

private:
    int startPoint, endPoint, currentPoint;
    unsigned char error;
    ::size_t viewSize;
    float *viewEntropy, minEntropy, maxEntropy;
};

#endif // QHEMISPHERE_H
