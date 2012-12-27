/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSWidget.h
 * @brief   QVSWidget class definition.
 * 
 * This file defines ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#ifndef QVSWIDGET_H
#define QVSWIDGET_H

#include <string>
#include <vector>
#include <list>

#include <gl/glew.h>
#include <cl/cl.h>

#include <QtOpenGL/QGLWidget>

#include "QVSSetting.h"

enum QMouseMode;
enum QEndianness;
enum QDataFormat;

class QKeyEvent;

class QVector3;
class QVector4;
class QHoverPoints;
class QCLProgram;
class QCLMemory;
class QVSControlPanel;

class QVSWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    QVSWidget(QWidget *parent = 0);
    ~QVSWidget();
    
    // Data file
    std::string dataFilePath, dataFileName, objectFileName;
    
    // Volumetric Data
    QEndianness endian;
    QDataFormat format;
    cl_uint4 volumeSize;
    cl_float timeInterval;
    cl_float4 boxSize, thickness;

    // Transfer Function
    std::vector<::size_t> transferFunctionSize;
    std::vector<unsigned char> transferFunctionData;
    
    // Memory Cache
    cl_uint cacheVolumeSize, cacheHistogramSize;
    std::vector<unsigned char> cacheVolumeData, cacheHistogramData, cacheVisibilityData, cacheEntropyData, cacheNoteworthinessData;
#ifdef __CL_ENABLE_DEBUG
    cl_uint cacheDebugSize;
    std::vector<unsigned char> cacheDebugData;
#endif
    float volumeMin, volumeMax, histogramMin, histogramMax;
    
    // Configuration
    cl_bool initialized;
    cl_float windowWidth, windowLevel;
    unsigned char error;
    QVSSetting settings;
    
    // OpenGL Context
    QMouseMode mouseMode;
    int mouseX, mouseY;
    std::vector<unsigned char> inverseViewMatrix;
    
    // OpenCL Context
    std::list<QCLProgram> clPrograms;
    std::vector<cl_device_id> clDevices;
    cl_context clContext;
    cl_command_queue clQueue;

    // View Entropy
    ::size_t viewSize;
    std::vector<cl_float> viewEntropy;
    
    QVSControlPanel* panel;
    
    unsigned char destroy();
    
    // Step 1 - init connections
    unsigned char initConnections(QVSControlPanel* panel);

    // Step 2 - init data
    unsigned char initData(const std::string &name);
    unsigned char parseDataFile(const std::string &name);
    unsigned char initConfigurations();
    
    // Step 3 - init context
    void initContext();
    unsigned char initOpenCL();
    unsigned char initArguments();
    unsigned char initPrograms();
    
    // Step 4 - message loop

    // slotMarkPoint
    void markPoint(int type);

    // slotUpdateTransferFunction
    unsigned char updateTransferFunction(QHoverPoints *controlPoints, int width);

    // resizeGL
    unsigned char updatePixelBuffer();

    // paintGL
    unsigned char computeHistograms(unsigned char enableComputingEntropy = GL_TRUE);
    unsigned char computeEntropy(const QVector4 &rotation, unsigned char paint = GL_TRUE);
    unsigned char drawVolume();
    void printFPS(unsigned long milliseconds);

    // slotConfigurations
    void saveSettings(const std::string &name = "settings.cfg");
    void loadSettings(const std::string &name = "settings.cfg");
    void saveViewEntropy(const std::string &name = "view_entropy.cfg");
    void loadViewEntropy(const std::string &name = "view_entropy.cfg");

    // keyPressEvent
    void printSettings();

signals:
    void signalHistogramInitialized(::size_t histogramSize, float *histogramData);
    void signalHistogramUpdated(unsigned int histogramID, float *histogramData);
    void signalNorthernViewEntropyInitialized(::size_t viewSize, float *viewEntropy, float minEntropy, float maxEntropy);
    void signalSouthernViewEntropyInitialized(::size_t viewSize, float *viewEntropy, float minEntropy, float maxEntropy);
    void signalNorthernViewPointMarked(int type, int offset);
    void signalSouthernViewPointMarked(int type, int offset);
    void signalViewEntropyUpdated(double entropy);
    void signalLoadViewEntropy();
    void signalSaveViewEntropy();

public slots:
    void slotUpdateTransferFunction(QHoverPoints *controlPoints, int width, unsigned char modified);
    void slotUpdateVolumeStepSize(int value);
    void slotUpdateVolumeOffset(int value);
    void slotUpdateVolumeScale(int value);
    void slotUpdateComputingEntropyState(bool value);
    void slotUpdateShadingState(bool value);
    void slotUpdateGaussian1DState(bool value);
    void slotUpdateGaussian2DState(bool value);
    void slotUpdateLightPositionX(double value);
    void slotUpdateLightPositionY(double value);
    void slotUpdateLightPositionZ(double value);
    void slotUpdateLightColor(const QColor &color);
    void slotUpdateLightDiffuseCoeff(double value);
    void slotUpdateLightAmbientCoeff(double value);
    void slotUpdateLightSpecularCoeff(double value);
    void slotUpdateMaterialShininess(int value);
    void slotLoadConfigurations();
    void slotSaveConfigurations();
    void slotComputeViewEntropy();
    void slotMarkStartPoint();
    void slotMarkEndPoint();

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);

private:
    std::string getOptions();
};

#endif // QVSWIDGET_H