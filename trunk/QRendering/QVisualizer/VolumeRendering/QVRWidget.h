/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRWidget.h
 * @brief   QVRWidget class definition.
 * 
 * This file defines the main process of Time-varying Volumetric Data Visualization Framework.
 *     The framework is a rendering pipeline base on OpenCL framework.
 *     The general process of rendering consists of three steps.
 *     1. Initialization
 *         The initialization step contains three small steps which are
 *         1) Connection Initialization
 *             This step makes connections between the foreground UIs and the background system.
 *         2) Data Initialization
 *             This step parses the data file.
 *         3) Context Initialization
 *             This step
 *                 checks the levels of support of OpenGL and OpenCL,
 *                 initializes the OpenGL and OpenCL context,
 *                 tries allocating the memory,
 *                 builds the required OpenCL programs,
 *                 and starts the rendering pipeline.
 *     2. Scheduling
 *         This step mainly focuses on pipeline scheduling which includes two parts:
 *             the master process is in charge of user interactions.
 *             other processes correspond to different stages in the pipeline.
 *         In our pipeline system five stages are supported.
 *         1) I/O
 *             In this stage, time-varying volumetric data are loaded from the disk to memory.
 *         2) Preprocessing
 *             In this stage, the volumetric data are locally preprocessed for the purpose of data formatting.
 *         3) Data Transmission
 *             In this stage, the formatted data are transmitted from CPU to GPU.
 *         4) GPU preprocessing
 *             In this stage, the data in GPU are used to compute other important features for rendering.
 *         5) Rendering
 *             In this stage, the data are visualized using OpenCL.
 *     3. Destroy
 *         The final step terminates all the threads and releases all the resources.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QVRWIDGET_H
#define QVRWIDGET_H

#include <vector>
#include <list>

#include <gl/glew.h>
#include <cl/cl.h>

#include <QtOpenGL/QGLWidget>

enum QMouseMode;
enum QEndianness;
enum QDataFormat;
enum StageState;

class QKeyEvent;
class QMutex;
class QWaitCondition;

class QVector3;
class QHoverPoints;
class QCLProgram;
class QCLMemory;
class QVRReader;
class QVRPreprocessor;
class QVRWriter;
class QVRSetting;
class QVRControlPanel;

class QVRWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    QVRWidget(QWidget *parent = 0);
    ~QVRWidget();
    
    // Data file
    std::string dataFilePath, dataFileName, objectFileName;
    
    // Volumetric Data
    QEndianness endian;
    QDataFormat format;
    cl_uint volumeOrigin, timeSteps;
    cl_uint4 volumeSize;
    cl_float timeInterval;
    cl_float4 boxSize, thickness;

    // Transfer Function
    std::vector<::size_t> transferFunctionSize;
    std::vector<unsigned char> transferFunctionData;
    
    // Memory Cache
    ::size_t cacheSize, clCacheSize, cacheVolumeSize, cacheHistogramSize;
    std::vector<StageState> cacheStatus;
    std::vector<cl_uint> cacheMapping, clCacheMapping;
    std::vector<unsigned char> cacheVolumeData, cacheHistogramData;
    QCLMemory* clCacheVolumeData;
#ifdef __CL_ENABLE_DEBUG
    std::vector<unsigned char> cacheDebugData;
#endif
    float valueMin, valueMax;
    
    // Configuration
    cl_bool initialized;
    cl_float windowWidth, windowLevel;
    unsigned char error;
    QVRSetting* settings;
    
    // OpenGL Context
    QMouseMode mouseMode;
    int mouseX, mouseY;
    std::vector<unsigned char> inverseViewMatrix;
    
    // OpenCL Context
    std::list<QCLProgram> clPrograms;
    std::vector<cl_device_id> clDevices;
    std::vector<::size_t> localSize, gridSize;
    cl_context clContext;
    cl_command_queue clQueue;
    
    // Pipeline
    QMutex *volumeMutex, *statusMutex;
    QWaitCondition *readingFinished, *preprocessingFinished, *writingFinished, *paintingFinished;
    QVRReader* reader;
    QVRPreprocessor* preprocessor;
    QVRWriter* writer;
    QVRControlPanel* panel;
    
    unsigned char destroy();
    
    // Step 1 - init connections
    unsigned char initConnections(QVRControlPanel* panel);

    // Step 2 - init data
    unsigned char initData(const std::string &name);
    unsigned char parseDataFile(const std::string &name);
    unsigned char initConfigurations();
    
    // Step 3 - init context
    void initContext();
    unsigned char initOpenCL();
    unsigned char initPipeline();
    unsigned char initArguments();
    unsigned char initPrograms();
    
    // Step 4 - message loop

    // slotUpdateTransferFunction
    unsigned char updateTransferFunction(QHoverPoints *controlPoints, int width);

    // slotUpdateTimeStep
    unsigned char updateVolume();

    // resizeGL
    unsigned char updatePixelBuffer();

    // paintGL
    unsigned char drawVolume();
    void printFPS(unsigned long milliseconds);

    // keyPressEvent
    void saveSettings();
    void loadSettings();
    void printSettings();

signals:
    void signalHistogramInitialized(float *histogramData, ::size_t histogramSize);
    void signalHistogramUpdated(unsigned int histogramID, float *histogramData);

public slots:
    unsigned char slotUpdateTransferFunction(QHoverPoints *controlPoints, int width);
    unsigned char slotUpdateStepSize();
    unsigned char slotUpdateVolumeOffset();
    unsigned char slotUpdateVolumeScale();
    unsigned char slotUpdateTimeStep();

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
};

#endif // QVRWIDGET_H