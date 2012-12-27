/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QTransferFunction1D.cpp
 * @brief   QTransferFunction1D class declaration.
 * 
 * This file declares the methods of the widget defined in QTransferFunction1D.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include "QTransferFunction1D.h"

QTransferFunction1D::QTransferFunction1D(QWidget *parent) : QWidget(parent),
    histogramData(0), histogramSize(0)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QPolygonF points(0);
    points.push_back(QPointF(0.0, this->height()));
    points.push_back(QPointF(this->width(), 0.0));

    hoverPoints = new QHoverPoints(this, QHoverPoints::CircleShape);
    hoverPoints->setConnectionType(QHoverPoints::LineConnection);
    hoverPoints->setPoints(points);
    hoverPoints->setPointLock(0, QHoverPoints::LockToLeft);
    hoverPoints->setPointLock(1, QHoverPoints::LockToRight);
    hoverPoints->setSortType(QHoverPoints::XSort);

    QVector<QColor> colors(0);
    colors.push_back(QColor::fromHslF(0.0f, 1.0f, 1.0f, 0.0f));
    colors.push_back(QColor::fromHslF(1.0f, 1.0f, 1.0f, 1.0f));
    hoverPoints->setColors(colors);

    connect(hoverPoints, SIGNAL(signalPointsChanged(unsigned char)), this, SLOT(slotUpdateTransferFunction(unsigned char)));
}

void QTransferFunction1D::drawHistogram()
{
    std::vector<float*>::iterator data = histogramData.begin();
    for (std::vector<::size_t>::iterator size = histogramSize.begin(); size != histogramSize.end(); size++, data++)
    {
        ::size_t histogramSize = *size;
        float* histogramData = *data;

        // Generate the histogram path
        qreal scale = this->height() * 0.8f;
        qreal stepSize = (float)this->width() / histogramSize;

        QPainterPath* histogramPath = new QPainterPath();
        histogramPath->moveTo(0.0, 0.0);
        histogramPath->lineTo(0.0, this->height() - *histogramData * scale);

        qreal x = -stepSize * 0.5, y = 0.0;
        for(int i = 0; i < histogramSize; i++)
        {
            x += stepSize;
            y = this->height() - *(histogramData++) * scale;
            histogramPath->lineTo(x, y);
        }
        histogramPath->lineTo(this->width(), histogramPath->currentPosition().y());
        histogramPath->lineTo(this->width(), 0.0);
        histogramPath->lineTo(0.0, 0.0);

        QPainter painter(this);
        QColor color(200, 200, 200, 125);
        painter.setBrush(color);
        painter.setPen(Qt::gray);
        painter.setRenderHints(QPainter::Antialiasing);
        painter.drawPath(*histogramPath);
        painter.end();
    }
}

void QTransferFunction1D::generateLinearGradient(QLinearGradient &linearGradient)
{
    float width = this->width();
    linearGradient.setStart(0.00, 0.00);
    linearGradient.setFinalStop(this->width(), 0.00);

    QPolygonF *points = &hoverPoints->points();
    QVector<QColor> *colors = &hoverPoints->colors();
    for (QVector<QPointF>::iterator i = points->begin();  i != points->end(); i++)
    {
        linearGradient.setColorAt(i->x() / width, colors->at(i - points->begin()));
    }
}

void QTransferFunction1D::drawColorBar()
{
    QPolygonF *points = &hoverPoints->points();
    if (points->size() < 2) return;

    QPainterPath* hoverPointPath = new QPainterPath();
    hoverPointPath->moveTo(0.0f, this->height());
    for (QVector<QPointF>::iterator i = points->begin();  i != points->end(); i++)
    {
        hoverPointPath->lineTo(*i);
    }
    hoverPointPath->lineTo(this->width(), this->height());

    //create the linear gradient.
    QLinearGradient linearGradient;
    this->generateLinearGradient(linearGradient);

    QPainter painter(this);
    painter.setBrush(linearGradient);
    painter.setPen(Qt::darkGray);
    painter.setRenderHints(QPainter::Antialiasing);
    painter.drawPath(*hoverPointPath);
    painter.end();
}

void QTransferFunction1D::slotInsertHistogram(::size_t histogramSize, float *histogramData)
{
    this->histogramSize.push_back(histogramSize);
    this->histogramData.push_back(histogramData);
    update();
}

void QTransferFunction1D::slotUpdateHistogram(unsigned int histogramID, float *histogramData)
{
    if (histogramID < this->histogramData.size() && histogramData)
        this->histogramData.at(histogramID) = histogramData;
    update();
}

void QTransferFunction1D::slotLoadConfigurations(std::string name)
{
    hoverPoints->load(name, this->size());
}

void QTransferFunction1D::slotSaveConfigurations(std::string name)
{
    hoverPoints->save(name, this->size());
}

void QTransferFunction1D::paintEvent(QPaintEvent *e)
{
    drawColorBar();
    drawHistogram();

    e->accept();
}

void QTransferFunction1D::resizeEvent(QResizeEvent *e)
{
    emit signalControlPointsChanged(hoverPoints, this->width(), GL_TRUE);
    
    e->accept();
}

void QTransferFunction1D::setAlpha()
{
    QPolygonF *points = &hoverPoints->points();
    QVector<QColor> *colors = &hoverPoints->colors();
    QVector<QPointF>::iterator point = points->begin();
    for (QVector<QColor>::iterator i = colors->begin(); i != colors->end(); i++)
    {
        i->setAlphaF(1.0f - (point++)->y() / this->height());
    }
}

void QTransferFunction1D::slotUpdateTransferFunction(unsigned char modified)
{
    setAlpha();
    emit signalControlPointsChanged(hoverPoints, this->width(), modified);
}