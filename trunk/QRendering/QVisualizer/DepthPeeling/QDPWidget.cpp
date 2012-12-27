/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QDPWidget.cpp
 * @brief   QDPWidget class declaration.
 * 
 * This file declares ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/06
 */

#include <gl/glew.h>
#include <cl/cl_gl.h>

#include <cmath>
#include <iomanip> 
#include <iostream>
#include <sstream>

#include <QKeyEvent>
#include <QDir>

#include "../utilities/QIO.h"
#include "../infrastructures/QHoverPoints.h"
#include "../infrastructures/QCLProgram.h"
#include "../infrastructures/QPipeline.h"
#include "../infrastructures/QProgram.h"
#include "../infrastructures/QTexture.h"
#include "../infrastructures/QModel.h"
#include "../infrastructures/QVTKModel.h"
#include "QDPSetting.h"
#include "QDPControlPanel.h"
#include "QDPWidget.h"

// [houtao]
#include "float.h"

const GLfloat QDPWidget::vertexScale = 1.0f / 250;
const GLuint QDPWidget::numPasses = 5;
const GLuint QDPWidget::imageWidth = 1024;
const GLuint QDPWidget::imageHeight = 768;
const GLenum QDPWidget::drawBuffers[2][6] =
{
    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5,
    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5
};

QDPWidget::QDPWidget(QWidget *parent)
    : QGLWidget(parent),
    dataFileName(), dataFilePath(), objectFileName(), // Data file
    endian(ENDIAN_LITTLE), volumeSize(), thickness(), modelScale(), sampleScale(), format(DATA_UNKNOWN), intenityLevel(256),
        valueMin(FLT_MAX), valueMax(-FLT_MAX), // Volumetric Data
    cacheHistogramSize(0), cacheHistogramData(0), cacheVolumeSize(0), cacheVolumeData(0), // Memory Cache
#ifdef __CL_ENABLE_DEBUG
    cacheDebugData(0),
#endif
    initialized(CL_FALSE), windowWidth(1.0f), windowLevel(0.5f), error(GL_FALSE), settings(new QDPSetting()), // Configuration
#ifdef __GL_ENABLE_DEBUG
    debugDepthBuffer(imageWidth * imageHeight * 2 * sizeof(GL_FLOAT)),
    debugColorBuffer(imageWidth * imageHeight * 4 * sizeof(GL_FLOAT)),
#endif
    mouseMode(MOUSE_ROTATE), mouseX(0), mouseY(0), glModels(0), glVTKModels(0), modelColor(4, 0.0f), glPrograms(0), glTextures(0),
    quadDisplayList(0), peelingSingleFBO(2, 0), // OpenGL Context
    clPrograms(), clDevices(), glVBO(0), clContext(0), clQueue(0) // OpenCL Context
{}

QDPWidget::~QDPWidget()
{
    this->destroy();
}

unsigned char QDPWidget::destroy()
{
    if (this->settings) delete this->settings;

    if (this->clQueue)
    {
        cl_int status = clReleaseCommandQueue(this->clQueue);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseCommandQueue()")) return GL_FALSE;
        this->clQueue = 0;
    }

    if (this->clContext)
    {
        cl_int status = clReleaseContext(this->clContext);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseContext()")) return GL_FALSE;
        this->clContext = 0;
    }

    return GL_TRUE;
}

// Step 1 - init connections
unsigned char QDPWidget::initConnections(QDPControlPanel* panel)
{
    this->panel = panel;

    const Ui::QDPControlPanel* ui = panel->getUI();
    connect(this, SIGNAL(signalHistogramInitialized(::size_t, float*)), ui->widgetEditor, SLOT(slotInsertHistogram(::size_t, float*)));
    connect(this, SIGNAL(signalHistogramUpdated(unsigned int, float*)), ui->widgetEditor, SLOT(slotUpdateHistogram(unsigned int, float*)));
    connect(ui->widgetEditor, SIGNAL(signalControlPointsChanged(QHoverPoints*, int)), this, SLOT(slotUpdateTransferFunction(QHoverPoints*, int)));
    connect(ui->horizontalSliderStepSize, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateStepSize(int)));
    connect(ui->horizontalSliderVolumeOffset, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateVolumeOffset(int)));
    connect(ui->horizontalSliderVolumeScale, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateVolumeScale(int)));
    connect(ui->horizontalSliderColor, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateAlpha(int)));
    connect(ui->horizontalSliderAlpha, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateAlpha(int)));

    settings->volumeStepSize = 1.0f / ui->horizontalSliderStepSize->value();
    settings->volumeOffset = ui->horizontalSliderVolumeOffset->value() * 0.01f;
    settings->volumeScale = ui->horizontalSliderVolumeScale->value() * 0.01f;

    getColor(modelColor.data());

    return GL_TRUE;
}

// Step 2 - init data
unsigned char QDPWidget::initData(const std::string &name)
{
    dataFileName = name;
    int position = dataFileName.find_last_of("\\");
    if (position == std::string::npos) position = dataFileName.find_last_of("/");
    if (position == std::string::npos) position = dataFileName.size() - 1;
    dataFilePath = dataFileName.substr(0, position + 1);

    if (error = !parseDataFile(dataFileName)) return GL_FALSE;
    
    return GL_TRUE;
}

unsigned char QDPWidget::parseDataFile(const std::string &name)
{
    std::string dataFileContent, line;
    if (!QIO::getFileContent(name, dataFileContent)) return GL_FALSE;

    std::stringstream data(dataFileContent, std::stringstream::in);
    unsigned char error = GL_FALSE;
    ::size_t position = std::string::npos;
    while (!data.eof())
    {
        getline(data, line);
        std::stringstream buffer(std::stringstream::in | std::stringstream::out);
        if ((position = line.find("ObjectFileName")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            objectFileName = line.substr(position + 1);
            QUtility::trim(objectFileName);
        }
        else if ((position = line.find("Resolution")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            buffer << line.substr(position + 1);
            unsigned int x = 0, y = 0, z = 0;
            buffer >> x >> y >> z;
            if (x <= 0 || y <= 0 || z <= 0)
            {
                error = GL_TRUE;
                break;
            }
            volumeSize.s[0] = x;
            volumeSize.s[1] = y;
            volumeSize.s[2] = z;
        }
        else if ((position = line.find("SliceThickness")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            buffer << line.substr(position + 1);
            float x = 0.0, y = 0.0, z = 0.0;
            buffer >> x >> y >> z;
            if (x <= 0.0 || y <= 0.0 || z <= 0.0)
            {
                error = GL_TRUE;
                break;
            }
            thickness.s[0] = x;
            thickness.s[1] = y;
            thickness.s[2] = z;
        }
        else if ((position = line.find("Format")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            if ((position = line.find("UCHAR")) != std::string::npos)
            {
                format = DATA_UCHAR;
            }
            else if ((position = line.find("USHORT")) != std::string::npos)
            {
                format = DATA_USHORT;
            }
            else if ((position = line.find("FLOAT")) != std::string::npos)
            {
                format = DATA_FLOAT;
            }
            else
            {
                std::cerr << " > ERROR: cannot process data other than of UCHAR and USHORT format." << std::endl;
                error = GL_TRUE;
            }
        }
        else if ((position = line.find("Window")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            buffer << line.substr(position + 1);
            buffer >> windowWidth >> windowLevel;
            if (windowWidth <= 0.0f)
            {
                error = GL_TRUE;
                break;
            }
        }
        else if ((position = line.find("Endian")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            if ((position = line.find("BIG-ENDIAN")) != std::string::npos)
            {
                endian = ENDIAN_BIG;
            }
            else if ((position = line.find("LITTLE-ENDIAN")) != std::string::npos)
            {
                endian = ENDIAN_LITTLE;
            }
            else
            {
                std::cerr << " > ERROR: cannot process endian other than of BIG-ENDIAN and LITTLE-ENDIAN format." << std::endl;
                error = GL_TRUE;
            }
        }
        else
        {
            std::cerr << " > WARNING: skipping line \"" << line << "\"." << std::endl;
        }
    }

    if (error)
    {
        std::cerr << " > ERROR: parsing \"" << line << "\"." << std::endl;
        return GL_FALSE;
    }

    return GL_TRUE;
}

// Step 3 - init context

void QDPWidget::initContext()
{
    if (error |= !QUtility::checkSupport()) return;
    if (error |= !initOpenGL()) return;
    if (error |= !initOpenCL()) return;
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    this->initialized = true;
}

unsigned char QDPWidget::initOpenGL()
{
    makeFullScreenQuad();

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_NORMALIZE);
    glDisable(GL_DEPTH_TEST);

    return GL_TRUE;
}

void QDPWidget::makeFullScreenQuad()
{
    quadDisplayList = glGenLists(1);
    glNewList(quadDisplayList, GL_COMPILE);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glBegin(GL_QUADS);
    {
        glVertex2f(0.0, 0.0); 
        glVertex2f(1.0, 0.0);
        glVertex2f(1.0, 1.0);
        glVertex2f(0.0, 1.0);
    }
    glEnd();
    glPopMatrix();

    glEndList();
}

unsigned char QDPWidget::initOpenCL()
{
    cl_int status = CL_SUCCESS;

    // Platform info
    cl_uint platformNumber(0);
    status = ::clGetPlatformIDs(0, NULL, &platformNumber);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetPlatformIDs()")) return GL_FALSE;

    std::vector<cl_platform_id> platforms(platformNumber);
    std::cout << " > INFO: getting platform information." << std::endl;
    status = clGetPlatformIDs(platformNumber, platforms.data(), NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetPlatformIDs()")) return GL_FALSE;
    
    cl_bool openclSupported = 0, openglSupported = 0, imageSupport = 0;
    std::string corporation = 0 ? "AMD" : "NVIDIA";
    std::vector<char> buffer(1024 * 4);
    std::vector<cl_context_properties> properties(0);
    for(std::vector<cl_platform_id>::iterator i = platforms.begin(); i != platforms.end(); i++)
    {
        status = clGetPlatformInfo(*i, CL_PLATFORM_VERSION, buffer.size() - 1, buffer.data(), NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetPlatformInfo()")) return GL_FALSE;
        std::string info(buffer.data());

        status = clGetPlatformInfo(*i, CL_PLATFORM_NAME, buffer.size() - 1, buffer.data(), NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetPlatformInfo()")) return GL_FALSE;
        std::string name(buffer.data());

        status = clGetPlatformInfo(*i, CL_PLATFORM_VENDOR, buffer.size() - 1, buffer.data(), NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetPlatformInfo()")) return GL_FALSE;
        std::string vendor(buffer.data());

        if (name.find(corporation) == std::string::npos) continue;

        openclSupported = info.find("OpenCL 1.") != std::string::npos;
        if (openclSupported)
        {
            // Device info
            cl_uint deviceNumber(0);
            status = clGetDeviceIDs(*i, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceNumber);
            if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetDeviceIDs()")) return GL_FALSE;

            std::vector<cl_device_id> devices(deviceNumber);
            status = clGetDeviceIDs(*i, CL_DEVICE_TYPE_ALL, deviceNumber, devices.data(), NULL);
            if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetDeviceIDs()")) return GL_FALSE;

            for (std::vector<cl_device_id>::iterator j = devices.begin(); j != devices.end(); j++)
            {
                status = clGetDeviceInfo(*j, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport), &imageSupport, NULL);
                if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetDeviceInfo()")) return GL_FALSE;
                
                status = clGetDeviceInfo(*j, CL_DEVICE_EXTENSIONS, buffer.size() - 1, buffer.data(), NULL);
                if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetDeviceInfo()")) return GL_FALSE;
                std::string extension(buffer.data());
                
                openglSupported = extension.find(GL_SHARING_EXTENSION) != std::string::npos;

                if (openglSupported && imageSupport)
                {
                    // Define OS-specific context properties and create the OpenCL context
                    #ifdef __APPLE__
                        CGLContextObj kCGLContext = CGLGetCurrentContext();
                        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
                        properties.push_back(CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
                        properties.push_back((cl_context_properties)kCGLShareGroup);
                    #else
                        #ifdef UNIX
                            properties.push_back(CL_GL_CONTEXT_KHR);
                            properties.push_back((cl_context_properties)glXGetCurrentContext());
                            properties.push_back(CL_GLX_DISPLAY_KHR);
                            properties.push_back((cl_context_properties)glXGetCurrentDisplay());
                            properties.push_back(CL_CONTEXT_PLATFORM);
                            properties.push_back((cl_context_properties)(*i));
                            clDevices.push_back(*j);
                        #else // Win32
                            properties.push_back(CL_GL_CONTEXT_KHR);
                            properties.push_back((cl_context_properties)wglGetCurrentContext());
                            properties.push_back(CL_WGL_HDC_KHR);
                            properties.push_back((cl_context_properties)wglGetCurrentDC());
                            properties.push_back(CL_CONTEXT_PLATFORM);
                            properties.push_back((cl_context_properties)(*i));
                            clDevices.push_back(*j);
                        #endif
                    #endif
                        
                    break;
                }
            }
            break;
        }
    }
    properties.push_back(0);
    if (!openclSupported || !openglSupported || !imageSupport)  return GL_FALSE;

    std::cout << " > INFO: creating a context sharing with OpenGL." << std::endl;
    
    clContext = clCreateContext(properties.data(), clDevices.size(), clDevices.data(), NULL, NULL, &status);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateContext()")) return GL_FALSE;

    clQueue = clCreateCommandQueue(clContext, clDevices.front(), 0, &status);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateCommandQueue()")) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QDPWidget::initConfigurations()
{
    cacheVolumeData.reserve(CACHE_VOLUME_SIZE);
    cacheVolumeData.resize(CACHE_VOLUME_SIZE);
    if (error = cacheVolumeData.size() != CACHE_VOLUME_SIZE)
    {
        std::cerr << " > ERROR: allocating volume memory(" << CACHE_VOLUME_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }

    cl_uint maxSize = 0;
    for (int i = 0; i < 3; i++)
        if (volumeSize.s[i] > maxSize) maxSize = volumeSize.s[i];
    if (maxSize == 0) return GL_FALSE;

    cacheVolumeSize = volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2];
    if (error = cacheVolumeData.size() < cacheVolumeSize * sizeof(float))
    {
        std::cerr << " > ERROR: limited volume memory allocated." << std::endl;
        return GL_FALSE;
    }

    cacheHistogramSize = NUMBER_HIS_ENTRIES;
    ::size_t size = cacheHistogramSize * sizeof(float);
    cacheHistogramData.reserve(size);
    cacheHistogramData.resize(size);
    if (error = cacheHistogramData.size() != size)
    {
        std::cerr << " > ERROR: allocating histogram memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }

#ifdef __CL_ENABLE_DEBUG
    cacheDebugData.resize(CACHE_CL_DEBUG_SIZE);
    if (error = cacheDebugData.size() != CACHE_CL_DEBUG_SIZE)
    {
        std::cerr << " > ERROR: allocating debug memory(" << CACHE_CL_DEBUG_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }
#endif
    return GL_TRUE;
}

unsigned char QDPWidget::initPrograms()
{
    GLuint status = CL_SUCCESS;

    printf("\nloading shaders...\n");

    glPrograms.push_front(QProgram("Init", "./glsl/DepthPeeling/"));
    std::list<QProgram>::iterator pInit = glPrograms.begin();
    pInit->shaders.push_back(QShader(pInit->path, "Vertex Shader", GL_VERTEX_SHADER, "dual_peeling_init_vertex.glsl"));
    pInit->shaders.push_back(QShader(pInit->path, "Fragment Shader", GL_FRAGMENT_SHADER, "dual_peeling_init_fragment.glsl"));
    if (!pInit->initialize()) return GL_FALSE;

    glPrograms.push_front(QProgram("Peel", "./glsl/DepthPeeling/"));
    std::list<QProgram>::iterator pPeel = glPrograms.begin();
    pPeel->shaders.push_back(QShader(pPeel->path, "Vertex Shader", GL_VERTEX_SHADER, "dual_peeling_peel_vertex.glsl"));
    pPeel->shaders.push_back(QShader(pPeel->path, "Fragment Shader", GL_FRAGMENT_SHADER, "dual_peeling_peel_fragment.glsl"));
    if (!pPeel->initialize()) return GL_FALSE;

    glPrograms.push_front(QProgram("Blend", "./glsl/DepthPeeling/"));
    std::list<QProgram>::iterator pBlend = glPrograms.begin();
    pBlend->shaders.push_back(QShader(pBlend->path, "Vertex Shader", GL_VERTEX_SHADER, "dual_peeling_blend_vertex.glsl"));
    pBlend->shaders.push_back(QShader(pBlend->path, "Fragment Shader", GL_FRAGMENT_SHADER, "dual_peeling_blend_fragment.glsl"));
    if (!pBlend->initialize()) return GL_FALSE;
    
    glPrograms.push_front(QProgram("Final", "./glsl/DepthPeeling/"));
    std::list<QProgram>::iterator pFinal = glPrograms.begin();
    pFinal->shaders.push_back(QShader(pFinal->path, "Vertex Shader", GL_VERTEX_SHADER, "dual_peeling_final_vertex.glsl"));
    pFinal->shaders.push_back(QShader(pFinal->path, "Fragment Shader", GL_FRAGMENT_SHADER, "dual_peeling_final_fragment.glsl"));
    if (!pFinal->initialize()) return GL_FALSE;

    glPrograms.push_front(QProgram("Volume Rendering", "./glsl/VolumeRendering/"));
    std::list<QProgram>::iterator pVolumeRendering = glPrograms.begin();
    pVolumeRendering->shaders.push_back(QShader(pVolumeRendering->path, "Vertex Shader", GL_VERTEX_SHADER, "ray_casting_vertex.glsl"));
    pVolumeRendering->shaders.push_back(QShader(pVolumeRendering->path, "Fragment Shader", GL_FRAGMENT_SHADER, "ray_casting_fragment.glsl"));
    if (!pVolumeRendering->initialize()) return GL_FALSE;
    
    glTextures.push_front(QTexture(GL_TEXTURE_3D, GL_R32F, GL_RED, GL_FLOAT, "Volume Data"));
    glTextures.push_front(QTexture(GL_TEXTURE_1D, GL_RGBA, GL_RGBA, GL_FLOAT, "Transfer Function Data"));

    return GL_TRUE;
}

unsigned char QDPWidget::initArguments()
{
    cl_int status = CL_SUCCESS;
    
    cl_uint voxelSize = 0;
    switch(format)
    {
    case DATA_UCHAR:
        voxelSize = 1;
        break;
    case DATA_USHORT:
        voxelSize = 2;
        break;
    case DATA_FLOAT:
        voxelSize = 4;
        break;
    }
    
    QIO::getFileData(dataFilePath + objectFileName, cacheVolumeData.data(), voxelSize * cacheVolumeSize);

    float volumeMin = FLT_MAX, volumeMax = FLT_MIN;
    QUtility::preprocess(cacheVolumeData.data(), cacheVolumeSize, format, endian, cacheHistogramSize, (float*)cacheHistogramData.data(), volumeMin, volumeMax);
    
    std::list<QTexture>::iterator pVolumeData = QTexture::find(glTextures, "Volume Data");
    if (!pVolumeData->initialize(std::vector<GLsizei>(volumeSize.s, volumeSize.s + 3), cacheVolumeData.data())) return GL_FALSE;

    emit(signalHistogramInitialized(cacheHistogramSize, (float*)cacheHistogramData.data()));

    float* pTransferFunction = settings->transferFunctionData.data();
    float scale = 1.0f / (NUMBER_TF_ENTRIES - 1);
    for (int i = 0; i < NUMBER_TF_ENTRIES; i++)
    {
        float value = i * scale;
        *(pTransferFunction++) = value;
        *(pTransferFunction++) = value;
        *(pTransferFunction++) = value;
        *(pTransferFunction++) = value;
    }
    
    std::list<QTexture>::iterator pTransferFunctionData = QTexture::find(glTextures, "Transfer Function Data");
    if (!pTransferFunctionData->initialize(settings->transferFunctionSize, settings->transferFunctionData.data())) return GL_FALSE;

    cl_float maxSize = 0.0;
    for (int i = 0; i < 3; i++)
    {
        modelScale.s[i] = volumeSize.s[i] * thickness.s[i];
        if (modelScale.s[i] > maxSize) maxSize = modelScale.s[i];
    }

    for (int i = 0; i < 3; i++)
    {
        modelScale.s[i] /= maxSize;
        sampleScale.s[i] = 1.0 / modelScale.s[i];
    }

    // E:/88Datasets/Tokyo/Fujita/
    // C:/Users/CGCADVIS/Desktop/DepthPeeling/

    std::string path("C:/Users/CGCADVIS/Desktop/DepthPeeling/");
    std::string filter(".bin");
    QDir dir(path.c_str());
    QFileInfoList files = dir.entryInfoList();
    std::string content;
    for (QFileInfoList::iterator i = files.begin(); i != files.end(); i++)
    {
        std::string file = std::string((const char *)i->fileName().toLocal8Bit());
        if (file.find(filter) != std::string::npos)
        {
            glModels.push_front(QModel((const char *)i->baseName().toLocal8Bit()));
            std::list<QModel>::iterator pModel = glModels.begin();
            pModel->loadBinaryFile((const char *)i->absoluteFilePath().toLocal8Bit());
            pModel->initialize();
            pModel->vertexScale = vertexScale;
            pModel->vertexTranslate.assign(3, 0.0f);
        }
    }

    std::list<QModel>::iterator mVolume = QModel::find(glModels, "Volume");
    if (mVolume != glModels.end())
    {
        mVolume->vertexScale = 1.0f;
        mVolume->vertexTranslate[2] = -0.1f;
    }
    /*
    glModels.push_front(QModel("Building"));
    std::list<QModel>::iterator pModel = glModels.begin();
    pModel->loadBinaryFile("E:/88Datasets/Tokyo/Fujita/building.bin");
    pModel->initialize();
    */
    std::vector<GLsizei> textureSize(2);
    textureSize[0] = imageWidth;
    textureSize[1] = imageHeight;
    
    glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_RGBA, GL_RGBA, GL_FLOAT, "Back Temp"));
    std::list<QTexture>::iterator tBackTemp = glTextures.begin();
    if (!tBackTemp->initialize(textureSize)) return GL_FALSE;
    glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_RGBA, GL_RGBA, GL_FLOAT, "Front Temp"));
    std::list<QTexture>::iterator tFrontTemp = glTextures.begin();
    if (!tFrontTemp->initialize(textureSize)) return GL_FALSE;
    glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_RGBA, GL_RGBA, GL_FLOAT, "Back Blender"));
    std::list<QTexture>::iterator tBackBlender = glTextures.begin();
    if (!tBackBlender->initialize(textureSize)) return GL_FALSE;
    
    glGenFramebuffers(peelingSingleFBO.size(), peelingSingleFBO.data());
    for (int i = 0; i < 2; i++)
    {
        std::stringstream name;
        name << "[" << i << "]";

        glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_FLOAT_RG32_NV, GL_RG, GL_FLOAT, "Depth Temp" + name.str()));
        std::list<QTexture>::iterator tDepthTemp = glTextures.begin();
        if (!tDepthTemp->initialize(textureSize)) return GL_FALSE;
        glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_FLOAT_RG32_NV, GL_RG, GL_FLOAT, "Depth_Blender" + name.str()));
        std::list<QTexture>::iterator tDepthBlender = glTextures.begin();
        if (!tDepthBlender->initialize(textureSize)) return GL_FALSE;
        glTextures.push_front(QTexture(GL_TEXTURE_RECTANGLE, GL_RGBA, GL_RGBA, GL_FLOAT, "Front Blender" + name.str()));
        std::list<QTexture>::iterator tFrontBlender = glTextures.begin();
        if (!tFrontBlender->initialize(textureSize)) return GL_FALSE;

        glBindFramebuffer(GL_FRAMEBUFFER, peelingSingleFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Depth_Blender" + name.str())->get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Depth Temp" + name.str())->get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Front Temp")->get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Back Temp")->get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Front Blender" + name.str())->get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_RECTANGLE, QTexture::find(glTextures, "Back Blender")->get(), 0);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glFramebufferTexture2D()"));
    }
    
    // Allocate render targets first
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    return GL_TRUE;
}

// Step 4 - message loop

// public slots
unsigned char QDPWidget::slotUpdateTransferFunction(QHoverPoints *controlPoints, int width)
{
    if (!this->initialized) return CL_FALSE;

    updateTransferFunction(controlPoints, width);
    updateGL();

    return GL_TRUE;
}

unsigned char QDPWidget::updateTransferFunction(QHoverPoints *controlPoints, int width)
{
    QPolygonF& points = controlPoints->points();
    QVector<QColor>& colors = controlPoints->colors();
    std::vector<float> alphas(colors.size());
    std::vector<float>::iterator p = alphas.begin();
    float scale = 1.0 / (BASE - 1.0f);
    for (QVector<QColor>::iterator i = colors.begin(); i!= colors.end(); i++)
        *(p++) = (::pow(BASE, i->alphaF()) - 1.0f) * scale;

    float *pointer = (float *)settings->transferFunctionData.data();
    float stepSize = (float)width / NUMBER_TF_ENTRIES;
    float x = stepSize * 0.5f;
    int size = settings->transferFunctionSize.at(0);
    int index = 0;
    for (int i = 0; i < size; ++i)
    {
        while (points.at(index + 1).x() <= x) index++;
        float ratio = float(x - points.at(index).x()) / (points.at(index + 1).x() - points.at(index).x());
        *(pointer++) = colors.at(index).redF() * (1 - ratio) + colors.at(index + 1).redF() * ratio;
        *(pointer++) = colors.at(index).greenF() * (1 - ratio) + colors.at(index + 1).greenF() * ratio;
        *(pointer++) = colors.at(index).blueF() * (1 - ratio) + colors.at(index + 1).blueF() * ratio;
        *(pointer++) = alphas.at(index) * (1 - ratio) + alphas.at(index + 1) * ratio;
        x += stepSize;
    }

    std::list<QTexture>::iterator pTransferFunctionData = QTexture::find(glTextures, "Transfer Function Data");
    if (!pTransferFunctionData->read(settings->transferFunctionData.data())) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateStepSize(int value)
{
    if (!this->initialized) return CL_FALSE;

    const Ui::QDPControlPanel* ui = panel->getUI();
    cl_float volumeStepSize = 1.0f / ui->horizontalSliderStepSize->value();
    if (volumeStepSize != settings->volumeStepSize)
    {
        settings->volumeStepSize = volumeStepSize;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateVolumeOffset(int value)
{
    if (!this->initialized) return CL_FALSE;

    const Ui::QDPControlPanel* ui = panel->getUI();
    cl_float volumeOffset = ui->horizontalSliderVolumeOffset->value() * 0.01f;
    if (volumeOffset != settings->volumeOffset)
    {
        settings->volumeOffset = volumeOffset;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateVolumeScale(int value)
{
    if (!this->initialized) return CL_FALSE;

    const Ui::QDPControlPanel* ui = panel->getUI();
    cl_float volumeScale = ui->horizontalSliderVolumeScale->value() * 0.01f;
    if (volumeScale != settings->volumeScale)
    {
        settings->volumeScale = volumeScale;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateTimeStep(int value)
{
    if (!this->initialized) return CL_FALSE;

    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateColor(int color)
{
    if (!this->initialized) return GL_FALSE;

    getColor(modelColor.data());
    updateGL();

    return GL_TRUE;
}

unsigned char QDPWidget::slotUpdateAlpha(int alpha)
{
    if (!this->initialized) return GL_FALSE;

    getColor(modelColor.data());
    updateGL();

    return GL_TRUE;
}

void QDPWidget::getColor(GLfloat* color)
{
    const Ui::QDPControlPanel* ui = panel->getUI();
    qreal alpha = (qreal)ui->horizontalSliderAlpha->value() / ui->horizontalSliderAlpha->maximum();
    qreal hue = (qreal)ui->horizontalSliderColor->value() / ui->horizontalSliderColor->maximum();
    QColor c = QColor::fromHsvF(hue, 0.8, 0.8, alpha);
    color[0] = c.redF();
    color[1] = c.greenF();
    color[2] = c.blueF();
    color[3] = c.alphaF();
}

// initializeGL
void QDPWidget::initializeGL()
{
    initContext();
}

// resizeGL
void QDPWidget::resizeGL(int w, int h)
{
    if (error) return;

    settings->width = w == 0 ? 1 : w;
    settings->height = h == 0 ? 1 : h;

	glViewport(0, 0,  settings->width, settings->height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)settings->width / settings->height, 0.5f, 10000.0f);

	updateGL();
}

// paintGL
void QDPWidget::paintGL()
{
    if (error) return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    glTranslatef(settings->viewTranslation.x, settings->viewTranslation.y, settings->viewTranslation.z);
	QVector4 rotation;
	QVector4::getAngleAxis(settings->viewRotation, rotation);
	glRotatef(rotation.w * 180.0 / PI, rotation.x, rotation.y, rotation.z);
	
    QDateTime tPaintGL = QDateTime::currentDateTime();
    
    glEnable(GL_BLEND);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);

    // ---------------------------------------------------------------------
    // 1. Initialize Min-Max Depth Buffer
    // ---------------------------------------------------------------------
    
    std::list<QModel>::iterator mVolume = QModel::find(glModels, "Volume");

    int currId = 1;
    int prevId = 1 - currId;
    glBindFramebuffer(GL_FRAMEBUFFER, peelingSingleFBO[currId]);

    // Render target 0 stores (-minDepth, maxDepth, alphaMultiplier)
    glDrawBuffer(drawBuffers[currId][0]);
    glClearColor(1.0f, 1.0f, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawBuffer(drawBuffers[currId][1]);
    glClearColor(FLT_MAX, FLT_MAX, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render targets 1 and 2 store the front and back colors
    // Clear to 0.0 and use MAX blending to filter written color
    // At most one front color and one back color can be written every pass
    glDrawBuffers(3, &drawBuffers[currId][2]);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // ---------------------------------------------------------------------
    // 2. Dual Depth Peeling + Blending
    // ---------------------------------------------------------------------

    // Since we cannot blend the back colors in the geometry passes,
    // we use another render target to do the alpha blending
    glDrawBuffer(drawBuffers[0][5]);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (int pass = 0; pass < numPasses; pass++)
    {
        currId = pass % 2;
        prevId = 1 - currId;

        std::stringstream currName, ptrvName;
        currName << "[" << currId << "]";
        ptrvName << "[" << prevId << "]";

        glBindFramebuffer(GL_FRAMEBUFFER, peelingSingleFBO[currId]);

        glDrawBuffer(drawBuffers[currId][0]);
        glClearColor(-1.0f, -1.0f, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawBuffer(drawBuffers[currId][1]);
        glClearColor(-FLT_MAX, -FLT_MAX, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawBuffers(3, &drawBuffers[currId][2]);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render target 0: RG32F MAX blending
        // Render target 1: RGBA MAX blending
        // Render target 2: RGBA MAX blending
        
        glDrawBuffers(5, &drawBuffers[currId][0]);
        glBlendEquation(GL_MAX);
        
        std::list<QProgram>::iterator pPeel = QProgram::find(glPrograms, "Peel");
        pPeel->activate();
        pPeel->setTexture(QTexture::find(glTextures, "Depth_Blender" + ptrvName.str()), "depthBlender", 0);
        pPeel->setTexture(QTexture::find(glTextures, "Front Blender" + ptrvName.str()), "frontBlender", 1);
        pPeel->setUniform("Color", (float*)modelColor.data(), 4);
        for (std::list<QModel>::iterator i = glModels.begin(); i != glModels.end(); i++)
            if (i != mVolume) i->paint();
        pPeel->deactivate();
        
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "Peel()")) return;
        
        // Full screen pass to alpha-blend the back color
        glDrawBuffer(drawBuffers[currId][5]);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        std::list<QProgram>::iterator pBlend = QProgram::find(glPrograms, "Blend");
        pBlend->activate();
        pBlend->setTexture(QTexture::find(glTextures, "Back Temp"), "temp", 0);
        glCallList(quadDisplayList);
        pBlend->deactivate();

        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "Blend()")) return;

        glDrawBuffer(drawBuffers[currId][3]);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawBuffers(2, &drawBuffers[currId][3]);
        glBlendEquation(GL_MAX);

        std::list<QProgram>::iterator pVolumeRendering = QProgram::find(glPrograms, "Volume Rendering");
        glEnable(GL_CULL_FACE);
        pVolumeRendering->activate();
        pVolumeRendering->setTexture(QTexture::find(glTextures, "Depth Temp" + ptrvName.str()), "prevDepth", 0);
        pVolumeRendering->setTexture(QTexture::find(glTextures, "Depth Temp" + currName.str()), "currDepth", 1);
        pVolumeRendering->setTexture(QTexture::find(glTextures, "Front Temp"), "currFrontBlender", 2);
        pVolumeRendering->setTexture(QTexture::find(glTextures, "Volume Data"), "volumeData", 3);
        pVolumeRendering->setTexture(QTexture::find(glTextures, "Transfer Function Data"), "transferFunctionData", 4);
        pVolumeRendering->setUniform("modelScale", (float*)modelScale.s, 3);
        pVolumeRendering->setUniform("sampleScale", (float*)sampleScale.s, 3);
        pVolumeRendering->setUniform("volumeOffset", (float*)&settings->volumeOffset, 1);
        pVolumeRendering->setUniform("volumeScale", (float*)&settings->volumeScale, 1);
        pVolumeRendering->setUniform("stepSize", (float*)&settings->volumeStepSize, 1);
        if (mVolume != glModels.end()) mVolume->paint();
        pVolumeRendering->deactivate();
        glDisable(GL_CULL_FACE);

        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "Volume Rendering()")) return;
        
#ifdef __GL_ENABLE_DEBUG
        glBindTexture(GL_TEXTURE_RECTANGLE, backBlender);
        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, GL_FLOAT, (GLvoid*)debugColorBuffer.data());
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glGetTexImage() - frontTemp")) return;
        float* dataPointer = (float*)debugColorBuffer.data() + 19800 * 4;
        cl_float4 vData = { dataPointer[0], dataPointer[1], dataPointer[2], dataPointer[3] };
        glBindTexture(GL_TEXTURE_RECTANGLE, depthTemp[currId]);
        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RG, GL_FLOAT, (GLvoid*)debugDepthBuffer.data());
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glGetTexImage() - depthTemp")) return;
        float* depthPointer = (float*)debugDepthBuffer.data() + 19800 * 2;
        cl_float2 vDepth = { depthPointer[0], depthPointer[1] };
        std::cerr << " > LOG:" << std::endl;
#endif
        
        // Full screen pass to alpha-blend the back color
        glDrawBuffer(drawBuffers[currId][5]);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pBlend->activate();
        pBlend->setTexture(QTexture::find(glTextures, "Back Temp"), "temp", 0);
        glCallList(quadDisplayList);
        pBlend->deactivate();

        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "Blend()")) return;
    }

    glDisable(GL_BLEND);

    // ---------------------------------------------------------------------
    // 3. Final Pass
    // ---------------------------------------------------------------------

    std::stringstream currName;
    currName << "[" << currId << "]";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);
    
    std::list<QProgram>::iterator pFinal = QProgram::find(glPrograms, "Final");
    pFinal->activate();
    pFinal->setTexture(QTexture::find(glTextures, "Front Blender" + currName.str()), "frontBlender", 0);
    pFinal->setTexture(QTexture::find(glTextures, "Back Blender"), "backBlender", 1);

    glCallList(quadDisplayList);
    pFinal->deactivate();
    
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "Final()")) return;
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_CULL_FACE);
    glColor3f(0.0f, 0.0f, 0.0f);
    for (std::list<QModel>::iterator i = glModels.begin(); i != glModels.end(); i++)
        i->paint();
    
    if (settings->enablePrintingFPS) QUtility::printFPS(tPaintGL.msecsTo(QDateTime::currentDateTime()), "paintGL()");

    // QMetaObject::invokeMethod(this, "updateGL", Qt::QueuedConnection);
}

void QDPWidget::mousePressEvent(QMouseEvent *event)
{
    mouseX = event->x();
    mouseY = event->y();

    if (event->buttons() & Qt::LeftButton)
    {
        mouseMode = MOUSE_ROTATE;
    }
    else if (event->buttons() & Qt::MidButton)
    {
        mouseMode = MOUSE_TRANSLATE;
    }
    else if (event->buttons() & Qt::RightButton)
    {
        mouseMode = MOUSE_DOLLY;
    }

    updateGL();
}

void QDPWidget::mouseMoveEvent(QMouseEvent *event)
{
    float dx = MOUSE_SCALE * (event->x() - mouseX) / (float)settings->width;
    float dy = MOUSE_SCALE * (event->y() - mouseY) / (float)settings->height;

    QVector3 mouse(dx, -dy, 0.0);
    QVector3 view(0.0, 0.0, -1.0);
    QVector3 rotateAxis = QVector3::normalize(QVector3::cross(mouse, view));

    switch (mouseMode)
    {
    case MOUSE_DOLLY:
        settings->viewTranslation.z += dy;
        break;
    case MOUSE_ROTATE:
        settings->viewRotation = QVector4::normalize(QVector4::fromAngleAxis(dx * dx + dy * dy, rotateAxis) * settings->viewRotation);
        break;
    case MOUSE_TRANSLATE:
        settings->viewTranslation = settings->viewTranslation + mouse;
        break;
    default:
        break;
    }

    mouseX = event->x();
    mouseY = event->y();

    updateGL();
}

void QDPWidget::wheelEvent(QWheelEvent *event)
{
    updateGL();
}

// keyPressEvent
void QDPWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Escape:
    case Qt::Key_Q:
        // close();
        break;
    case Qt::Key_S:
        saveSettings();
        break;
    case Qt::Key_L:
        loadSettings();
        break;
    case Qt::Key_P:
        printSettings();
        break;
    case Qt::Key_Space:
        settings->enablePrintingFPS = !settings->enablePrintingFPS;
        break;
    }

    updateGL();
}

void QDPWidget::saveSettings()
{}

void QDPWidget::loadSettings()
{}

void QDPWidget::printSettings()
{
    std::cerr << " > LOG: print frames per second " << settings->enablePrintingFPS << "." << std::endl;
    std::cerr << " > LOG: light direction (" << settings->lightDirection.x << ", " << settings->lightDirection.y << ", " << settings->lightDirection.z << std::endl;
    std::cerr << " > LOG: view rotation (" << settings->viewRotation.x << ", " << settings->viewRotation.y << ", " << settings->viewRotation.z << ", " << settings->viewRotation.w << ")." << std::endl;
    std::cerr << " > LOG: view translation (" << settings->viewTranslation.x << ", " << settings->viewTranslation.y << ", " << settings->viewTranslation.z << ")." << std::endl;
    std::cerr << " > LOG: window size (" << settings->width << ", " << settings->height << ")." << std::endl;
}
