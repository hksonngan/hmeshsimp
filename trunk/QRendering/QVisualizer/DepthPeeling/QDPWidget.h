/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPWidget.h
 * @brief   QDPWidget class definition.
 * 
 * This file defines the main process of Marching Cubes Algorithm.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#ifndef QDPWIDGET
#define QDPWIDGET

#include <vector>
#include <list>

#include <gl/glew.h>
#include <cl/cl.h>

#include <QtOpenGL/QGLWidget>

enum QMouseMode;
enum QDataFormat;
enum QEndianness;

class QKeyEvent;

class QVector3;
class QHoverPoints;
class QModel;
class QVTKModel;
class QProgram;
class QTexture;
class QCLProgram;
class QCLMemory;
class QDPSetting;
class QDPControlPanel;

class QDPWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    QDPWidget(QWidget *parent = 0);
    ~QDPWidget();
    
    // Data file
    std::string dataFilePath, dataFileName, objectFileName;
    
    // Volumetric Data
    QEndianness endian;
    QDataFormat format;
    cl_uint intenityLevel;
    cl_uint4 volumeSize;
    cl_float4 thickness, modelScale, sampleScale;

    // Memory Cache
    ::size_t cacheVolumeSize, cacheHistogramSize;
    std::vector<unsigned char> cacheVolumeData, cacheHistogramData;
#ifdef __CL_ENABLE_DEBUG
    std::vector<unsigned char> cacheDebugData;
#endif
    float valueMin, valueMax;
    
    // Configuration
    cl_bool initialized;
    cl_float windowWidth, windowLevel;
    unsigned char error;
    QDPSetting* settings;
    
    // OpenGL Context
    std::list<QProgram> glPrograms;
    std::list<QTexture> glTextures;
    std::list<QModel> glModels;
    std::list<QVTKModel> glVTKModels;

    QMouseMode mouseMode;
    int mouseX, mouseY;
    std::vector<GLfloat> modelColor;

    static const GLfloat vertexScale;
    static const GLuint numPasses;
    static const GLuint imageWidth;
    static const GLuint imageHeight;
    static const GLenum drawBuffers[2][6];

    GLuint quadDisplayList;
    std::vector<GLuint> peelingSingleFBO;
#ifdef __GL_ENABLE_DEBUG
    std::vector<GLubyte> debugDepthBuffer;
    std::vector<GLubyte> debugColorBuffer;
#endif

    // OpenCL Context
    std::list<QCLProgram> clPrograms;
    std::vector<cl_device_id> clDevices;
    cl_mem clVBO;
    cl_GLuint glVBO;
    cl_context clContext;
    cl_command_queue clQueue;
    
    // Widget
    QDPControlPanel* panel;
    
    unsigned char destroy();
    
    // Step 1 - init connections
    unsigned char initConnections(QDPControlPanel* panel);

    // Step 2 - init data
    unsigned char initData(const std::string &name);
    unsigned char parseDataFile(const std::string &name);
    
    // Step 3 - init context
    void initContext();
    unsigned char initOpenGL();
    unsigned char initOpenCL();
    unsigned char initConfigurations();
    unsigned char initPrograms();
    unsigned char initArguments();
    void makeFullScreenQuad();
    
    // Step 4 - message loop

    // slotUpdateTransferFunction
    unsigned char updateTransferFunction(QHoverPoints *controlPoints, int width);
    
    // paintGL
    void drawModel(const std::list<QModel>::iterator& pModel);
    void printFPS(unsigned long milliseconds);

    // keyPressEvent
    void saveSettings();
    void loadSettings();
    void printSettings();

    // others
    void getColor(GLfloat* color);

signals:
    void signalHistogramInitialized(::size_t histogramSize, float *histogramData);
    void signalHistogramUpdated(unsigned int histogramID, float *histogramData);

public slots:
    unsigned char slotUpdateTransferFunction(QHoverPoints *controlPoints, int width);
    unsigned char slotUpdateStepSize(int value);
    unsigned char slotUpdateVolumeOffset(int value);
    unsigned char slotUpdateVolumeScale(int value);
    unsigned char slotUpdateTimeStep(int value);
    unsigned char slotUpdateColor(int value);
    unsigned char slotUpdateAlpha(int value);

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