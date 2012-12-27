/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMCWidget.h
 * @brief   QMCWidget class definition.
 * 
 * This file defines the main process of Marching Cubes Algorithm.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#ifndef QMCWIDGET
#define QMCWIDGET

#include <vector>
#include <list>

#include <gl/glew.h>
#include <cl/cl.h>

#include <QtOpenGL/QGLWidget>

#include "QMCSetting.h"

enum QMouseMode;
enum QDataFormat;
enum QEndianness;

class QKeyEvent;

class QVector3;
class QHoverPoints;
class QCLProgram;
class QCLMemory;
class QMCControlPanel;

class QMCWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    QMCWidget(QWidget *parent = 0);
    ~QMCWidget();
    
    // Data file
    std::string dataFilePath, dataFileName, objectFileName;
    
    // Volumetric Data
    QEndianness endian;
    QDataFormat format;
    cl_uint intenityLevel, hpSize, hpHeight;
    cl_uint4 volumeSize;
    cl_float4 thickness;

    // Memory Cache
    ::size_t cacheVolumeSize;
    std::vector<unsigned char> cacheVolumeData;
#ifdef __CL_ENABLE_DEBUG
    std::vector<unsigned char> cacheDebugData;
#endif
    float valueMin, valueMax;
    
    // Configuration
    cl_bool initialized;
    cl_float windowWidth, windowLevel;
    unsigned char error;
    QMCSetting settings;
    
    // OpenGL Context
    QMouseMode mouseMode;
    int mouseX, mouseY;
    unsigned int bufferSize;
	unsigned int totalNumber;
    unsigned int isoSurfaceGenerated;
	float isoValue;
    
    // OpenCL Context
    std::list<QCLProgram> clPrograms;
    std::vector<cl_device_id> clDevices;
    cl_mem clVBO;
    cl_GLuint glVBO;
    cl_context clContext;
    cl_command_queue clQueue;
    
    // Widget
    QMCControlPanel* panel;
    
    unsigned char destroy();
    
    // Step 1 - init connections
    unsigned char initConnections(QMCControlPanel* panel);

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

    // paintGL
	unsigned char classifyCubes();
	unsigned char constructHP();
    unsigned char traverseHP(cl_uint size);

    void printFPS(unsigned long milliseconds);

    // keyPressEvent
    void saveSettings();
    void loadSettings();
    void printSettings();

public slots:
    unsigned char slotUpdateIsoValue();

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
};

#endif // QVRWIDGET