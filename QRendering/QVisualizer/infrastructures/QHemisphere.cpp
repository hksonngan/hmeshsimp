/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QHemisphere.cpp
 * @brief   QHemisphere class declaration.
 * 
 * This file declares the methods of the widget defined in QHemisphere.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/26
 */

#include <algorithm>

#include <QPaintEvent>
#include <QPainter>

#include "../utilities/QUtility.h"
#include "QHemisphere.h"

// [houtao]
#include "float.h"

QHemisphere::QHemisphere(QWidget *parent)
    : QWidget(parent),
    error(0), viewSize(0), viewEntropy(NULL), minEntropy(FLT_MAX), maxEntropy(0.0f),
    startPoint(-1), endPoint(-1), currentPoint(-1)
{}

QHemisphere::~QHemisphere()
{}

void QHemisphere::drawSquare(QPainter &painter, const QColor& pen, const QColor& brush, const qreal &x, const qreal &y, const qreal &size)
{
    painter.setPen(pen);
    painter.setBrush(brush);
    painter.drawRect(QRectF(x, y, size, size));
}

void QHemisphere::drawCircle(QPainter &painter, const QColor& pen, const QColor& brush, const qreal &x, const qreal &y, const qreal &radius)
{
    painter.setPen(pen);
    painter.setBrush(brush);
    painter.drawEllipse(QPointF(x, y), radius, radius);
}

void QHemisphere::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    
    QPointF center(width() * 0.5, height() * 0.5);
    qreal radius = std::min(width() * 0.5, height() * 0.5) - 1;
    painter.setPen(Qt::transparent);
    painter.setBrush(Qt::lightGray);
    painter.drawEllipse(center, radius, radius);

    if (viewSize > 0 && viewEntropy)
    {
        QPointF corner(center.x() - radius, center.y() - radius);
        qreal pointSize = radius * 2.0 / (viewSize - 1);
        qreal entropyScale = maxEntropy - minEntropy < EPSILON ? 1.0 : 1.0 / (maxEntropy - minEntropy);
        for (int y = 0; y < viewSize; y++)
            for (int x = 0; x < viewSize; x++)
            {
                float entropy = (viewEntropy[x + y * viewSize] - minEntropy) * entropyScale;
                if (entropy > 0.0f)
                    drawSquare(painter, Qt::transparent, QColor::fromRgbF(entropy, entropy, entropy), corner.x() + x * pointSize, corner.y() + y * pointSize, pointSize);
            }

        if (startPoint >= 0)
        {
            int x(startPoint % viewSize), y(startPoint / viewSize);
            drawCircle(painter, Qt::red, Qt::transparent, corner.x() + (x + 0.5) * pointSize, corner.y() + (y + 0.5) * pointSize, pointSize * 2);
        }
        if (endPoint >= 0)
        {
            int x(endPoint % viewSize), y(endPoint / viewSize);
            drawCircle(painter, Qt::blue, Qt::transparent, corner.x() + (x + 0.5) * pointSize, corner.y() + (y + 0.5) * pointSize, pointSize * 2);
        }
        if (currentPoint >= 0)
        {
            int x(currentPoint % viewSize), y(currentPoint / viewSize);
            drawCircle(painter, Qt::green, Qt::transparent, corner.x() + (x + 0.5) * pointSize, corner.y() + (y + 0.5) * pointSize, pointSize * 2);
        }
    }
    
    painter.setPen(Qt::black);
    painter.setBrush(Qt::transparent);
    painter.drawEllipse(center, radius, radius);

    event->accept();
}

void QHemisphere::slotInitViewEntropy(::size_t viewSize, float *viewEntropy, float minEntropy, float maxEntropy)
{
    if (viewSize > 0 && viewEntropy)
    {
        this->viewSize = viewSize;
        this->viewEntropy = viewEntropy;
        this->minEntropy = minEntropy;
        this->maxEntropy = maxEntropy;
    }
    update();
}

void QHemisphere::slotUpdateViewEntropy(double entropy)
{
    if (viewEntropy && entropy > 0.0)
    {
        if (entropy > maxEntropy) maxEntropy = entropy;
        if (entropy < minEntropy) minEntropy = entropy;
    }
    update();
}

void QHemisphere::slotMarkePoint(int type, int offset)
{
    switch (type)
    {
    case 0:
        startPoint = offset;
    	break;
    case 1:
        endPoint = offset;
        break;
    case 2:
        currentPoint = offset;
        break;
    default:
        break;
    }
}