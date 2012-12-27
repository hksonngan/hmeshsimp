/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRWidget.cpp
 * @brief   QVRWidget class declaration.
 * 
 * This file declares the methods of the main process of TVDVF defined in QVRWidget.h.
 *  
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <gl/glew.h>
#include <cl/cl_gl.h>

#include <cmath>
#include <iomanip> 
#include <iostream>
#include <sstream>

#include <QKeyEvent>
#include <QMutex>
#include <QWaitCondition>

#include "../utilities/QIO.h"
#include "../infrastructures/QHoverPoints.h"
#include "../infrastructures/QCLProgram.h"
#include "QVRPipeline.h"
#include "QVRSetting.h"
#include "QVRFeatureExtraction.h"
#include "QVRControlPanel.h"
#include "QVRWidget.h"

// [houtao]
#include "float.h"

QVRWidget::QVRWidget(QWidget *parent)
    : QGLWidget(parent),
    dataFileName(), dataFilePath(), objectFileName(), // Data file
    endian(ENDIAN_LITTLE), volumeSize(), volumeOrigin(0), thickness(), boxSize(), timeInterval(0), timeSteps(1), format(DATA_UNKNOWN),
        valueMin(FLT_MAX), valueMax(-FLT_MAX), // Volumetric Data
    transferFunctionSize(1, NUMBER_TF_ENTRIES), transferFunctionData(NUMBER_TF_ENTRIES * 4 * sizeof(float)),  // Transfer Function
    cacheSize(0), clCacheSize(0), cacheStatus(0), cacheMapping(0), clCacheMapping(0), cacheVolumeSize(0),
        cacheHistogramSize(0), cacheVolumeData(0), cacheHistogramData(0), clCacheVolumeData(NULL), // Memory Cache
#ifdef __CL_ENABLE_DEBUG
    cacheDebugData(0),
#endif
    initialized(CL_FALSE), windowWidth(1.0f), windowLevel(0.5f), error(GL_FALSE), settings(new QVRSetting()), // Configuration
    mouseMode(MOUSE_ROTATE), mouseX(0), mouseY(0), inverseViewMatrix(16 * sizeof(float)), // OpenGL Context
    clPrograms(), clDevices(), localSize(2, ::size_t(16)), gridSize(2, ::size_t(512)), clContext(0), clQueue(0), // OpenCL Context
    reader(new QVRReader(this)), preprocessor(new QVRPreprocessor(this)), writer(new QVRWriter(this)), statusMutex(new QMutex()), volumeMutex(NULL), 
        readingFinished(new QWaitCondition()), preprocessingFinished(new QWaitCondition()),
        writingFinished(new QWaitCondition()), paintingFinished(new QWaitCondition()) // Pipeline
{}

QVRWidget::~QVRWidget()
{
    this->destroy();
}

unsigned char QVRWidget::destroy()
{
    reader->terminate();
    reader->wait();

    preprocessor->terminate();
    preprocessor->wait();

    writer->terminate();
    writer->wait();
    
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

    if (clCacheVolumeData) delete clCacheVolumeData;

    if (reader) delete reader;
    if (preprocessor) delete preprocessor;
    if (writer) delete writer;

    if (statusMutex) delete statusMutex;
    if (volumeMutex) delete[] volumeMutex;

    if (readingFinished) delete readingFinished;
    if (preprocessingFinished) delete preprocessingFinished;
    if (paintingFinished) delete paintingFinished;

    return GL_TRUE;
}

// Step 1 - init connections
unsigned char QVRWidget::initConnections(QVRControlPanel* panel)
{
    this->panel = panel;

    const Ui::QVRControlPanel* ui = panel->getUI();
    connect(this, SIGNAL(signalHistogramInitialized(::size_t, float*)), ui->widgetEditor, SLOT(slotInsertHistogram(::size_t, float*)));
    connect(this, SIGNAL(signalHistogramUpdated(unsigned int, float*)), ui->widgetEditor, SLOT(slotUpdateHistogram(unsigned int, float*)));
    connect(ui->widgetEditor, SIGNAL(signalControlPointsChanged(QHoverPoints*, int)), this, SLOT(slotUpdateTransferFunction(QHoverPoints*, int)));
    connect(ui->horizontalSliderStepSize, SIGNAL(sliderReleased()), this, SLOT(slotUpdateStepSize()));
    connect(ui->horizontalSliderVolumeOffset, SIGNAL(sliderReleased()), this, SLOT(slotUpdateVolumeOffset()));
    connect(ui->horizontalSliderVolumeScale, SIGNAL(sliderReleased()), this, SLOT(slotUpdateVolumeScale()));
    connect(ui->horizontalSliderTimeStep, SIGNAL(sliderReleased()), this, SLOT(slotUpdateTimeStep()));

    settings->volumeStepSize = 1.0f / ui->horizontalSliderStepSize->value();
    settings->volumeOffset = ui->horizontalSliderVolumeOffset->value() * 0.01f;
    settings->volumeScale = ui->horizontalSliderVolumeScale->value() * 0.01f;

    return GL_TRUE;
}

// Step 2 - init data
unsigned char QVRWidget::initData(const std::string &name)
{
    dataFileName = name;
    int position = dataFileName.find_last_of("\\");
    if (position == std::string::npos) position = dataFileName.find_last_of("/");
    if (position == std::string::npos) position = dataFileName.size() - 1;
    dataFilePath = dataFileName.substr(0, position + 1);

    if (error = !parseDataFile(dataFileName)) return GL_FALSE;
    
    return GL_TRUE;
}

unsigned char QVRWidget::parseDataFile(const std::string &name)
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
        else if ((position = line.find("TimeSteps")) != std::string::npos)
        {
            if ((position = line.find(':')) == std::string::npos)
            {
                error = GL_TRUE;
                break;
            }
            buffer << line.substr(position + 1);
            buffer >> timeSteps >> timeInterval;
            if (timeSteps <= 0 || timeInterval <= 0.0f)
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
void QVRWidget::initContext()
{
    if (error |= !QUtility::checkSupport()) return;
    if (error |= !initOpenCL()) return;
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initPipeline()) return;
    if (error |= !initArguments()) return;
    /*
    QVRFeatureExtracation::extractFeatures(clContext, clPrograms, clQueue, &volumeData[0], volumeSize, voxelSize);
    */
    this->initialized = true;
}

unsigned char QVRWidget::initOpenCL()
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

unsigned char QVRWidget::initConfigurations()
{
    float actualSize[3];
    float maxActualSize = 0.0;
    for (int i = 0; i < 3; i++)
    {
        actualSize[i] = volumeSize.s[i] * thickness.s[i];
        if (actualSize[i] > maxActualSize) maxActualSize = actualSize[i];
    }

    for (int i = 0; i < 3; i++)
    {
        boxSize.s[i]= actualSize[i] / maxActualSize;
    }

    settings->volumeScale = windowWidth < EPSILON ? 1.0f : 1.0f / windowWidth;
    settings->volumeOffset = windowWidth * 0.5f - windowLevel;

    panel->getUI()->horizontalSliderTimeStep->setMaximum(timeSteps - 1);

    cacheVolumeData.reserve(CACHE_VOLUME_SIZE);
    cacheVolumeData.resize(CACHE_VOLUME_SIZE);
    if (error = cacheVolumeData.size() != CACHE_VOLUME_SIZE)
    {
        std::cerr << " > ERROR: allocating volume memory(" << CACHE_VOLUME_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }
    cacheVolumeSize = volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2];
    cacheSize = cacheVolumeData.size() / (cacheVolumeSize * sizeof(float));
    if (error = cacheSize == 0)
    {
        std::cerr << " > ERROR: limited volume memory allocated." << std::endl;
        return GL_FALSE;
    }

    clCacheVolumeData = new QCLMemory("CL Volume Cache", QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_ONLY);
    if (error = !clCacheVolumeData->initialize(clContext, std::vector<size_t>(1, CACHE_CL_VOLUME_SIZE), cacheVolumeData.data()))
    {
        std::cerr << " > ERROR: allocating gpu volume memory(" << CACHE_CL_VOLUME_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }
    clCacheSize = clCacheVolumeData->size.at(0) / (cacheVolumeSize * sizeof(float));
    if (error = clCacheSize == 0 || clCacheSize > cacheSize)
    {
        std::cerr << " > ERROR: limited gpu volume memory allocated." << std::endl;
        return GL_FALSE;
    }
    clCacheMapping.resize(clCacheSize);
    for (int i = 0; i < clCacheSize; i++)
    {
        clCacheMapping.at(i) = i;
    }

    cacheStatus.resize(cacheSize);
    cacheMapping.resize(cacheSize);
    for (int i = 0; i < cacheSize; i++)
    {
        cacheStatus.at(i) = QCL_INITIALIZED;
        cacheMapping.at(i) = i;
    }
    volumeMutex = new QMutex[cacheSize];

    cacheHistogramSize = NUMBER_HIS_ENTRIES;
    ::size_t size = cacheSize * cacheHistogramSize * sizeof(float);
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
        std::cerr << " > ERROR: allocating histogram memory(" << CACHE_CL_DEBUG_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }
#endif
    return GL_TRUE;
}

unsigned char QVRWidget::initPrograms()
{
    cl_int status = CL_SUCCESS;
    
    clPrograms.push_front(QCLProgram("Volume Rendering", "./cl/VoulmeRendering/"));
    std::list<QCLProgram>::iterator pVolumeRendering = clPrograms.begin();
    pVolumeRendering->kernels.push_front(QCLKernel(pVolumeRendering->path, "Volume Render", "volumeRendering", "volume_render.cl"));
    std::list<QCLKernel>::iterator kVolumeRender = pVolumeRendering->kernels.begin();
    if (!pVolumeRendering->initialize(clContext, clDevices)) return GL_FALSE;

    pVolumeRendering->memories.push_front(QCLMemory("Inverse View Matrix", QCLMemory::QCL_BUFFER, CL_TRUE, CL_FALSE, CL_MEM_READ_ONLY));
    std::list<QCLMemory>::iterator mInverseViewMatrix = pVolumeRendering->memories.begin();
    if (!mInverseViewMatrix->initialize(clContext, std::vector<::size_t>(1, inverseViewMatrix.size()), inverseViewMatrix.data())) return GL_FALSE;
    status = clSetKernelArg(kVolumeRender->get(), 0, sizeof(cl_mem), &mInverseViewMatrix->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    pVolumeRendering->memories.push_front(QCLMemory("Pixel Buffer", QCLMemory::QCL_BUFFERGL, CL_TRUE, CL_TRUE, CL_MEM_WRITE_ONLY));
    std::list<QCLMemory>::iterator mPixelBuffer = pVolumeRendering->memories.begin();
    if (!mPixelBuffer->initialize(clContext, std::vector<::size_t>(1, settings->width * settings->height * sizeof(GLubyte) * 4))) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 1, sizeof(cl_mem), &mPixelBuffer->get());
    status |= clSetKernelArg(kVolumeRender->get(), 2, sizeof(cl_mem), &clCacheVolumeData->get());
    status |= clSetKernelArg(kVolumeRender->get(), 4, sizeof(cl_uint4), &volumeSize);
    status |= clSetKernelArg(kVolumeRender->get(), 12, sizeof(cl_float4), &boxSize);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
    
    cl_image_format fTransferFunction;
    fTransferFunction.image_channel_order = CL_RGBA;
    fTransferFunction.image_channel_data_type = CL_FLOAT;
    pVolumeRendering->memories.push_front(QCLMemory("Transfer Function", QCLMemory::QCL_IMAGE2D, CL_FALSE, CL_FALSE, CL_MEM_READ_ONLY, fTransferFunction));
    std::list<QCLMemory>::iterator mTransferFunction = pVolumeRendering->memories.begin();
    if (!mTransferFunction->initialize(clContext, std::vector<::size_t>(settings->transferFunctionSize.begin(), settings->transferFunctionSize.end()), transferFunctionData.data())) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 5, sizeof(cl_mem), &mTransferFunction->get());
    status |= clSetKernelArg(kVolumeRender->get(), 6, sizeof(cl_sampler), &mTransferFunction->getSampler());

#ifdef __CL_ENABLE_DEBUG
    pVolumeRendering->memories.push_front(QCLMemory("Debug Output", QCLMemory::QCL_BUFFER, CL_FALSE, CL_TRUE, CL_MEM_WRITE_ONLY));
    std::list<QCLMemory>::iterator mDebugOutput = pVolumeRendering->memories.begin();
    if (!mDebugOutput->initialize(clContext, std::vector<::size_t>(1, CACHE_CL_DEBUG_SIZE), cacheDebugData.data())) return GL_FALSE;
    status = clSetKernelArg(kVolumeRender->get(), 13, sizeof(cl_mem), &mDebugOutput->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
#endif
    //
    /*
    clPrograms.push_front(QCLProgram("Feature Extraction", "./cl/Feature Extraction/"));
    std::list<QCLProgram>::iterator pFeatureExtraction = clPrograms.begin();
    pFeatureExtraction->kernels.push_front(QCLKernel(pFeatureExtraction->path, "Feature Extractor", "entrance_feature_extractor", "feature_extractor.cl"));
    std::list<QCLKernel>::iterator kFeatureExtractor = pFeatureExtraction->kernels.begin();
    if (!pFeatureExtraction->initialize(clContext, clDevices)) return GL_FALSE;
    */
    return GL_TRUE;
}

unsigned char QVRWidget::initPipeline()
{
    std::string fileNamePrefix(dataFilePath + objectFileName);
    reader->init(fileNamePrefix);
    reader->start();
    preprocessor->start();
    writer->start();

    return GL_TRUE;
}

unsigned char QVRWidget::initArguments()
{
    cl_int status = CL_SUCCESS;

    if (!updateVolume()) return GL_FALSE;

    return GL_TRUE;
}

// Step 4 - message loop

// public slots
unsigned char QVRWidget::slotUpdateTransferFunction(QHoverPoints *controlPoints, int width)
{
    if (!this->initialized) return CL_FALSE;

    updateTransferFunction(controlPoints, width);
    updateGL();

    return GL_TRUE;
}

unsigned char QVRWidget::updateTransferFunction(QHoverPoints *controlPoints, int width)
{
    QPolygonF &points = controlPoints->points();
    QVector<QColor> &colors = controlPoints->colors();
    std::vector<float> alphas(colors.size());
    std::vector<float>::iterator p = alphas.begin();
    float scale = 1.0 / (BASE - 1.0f);
    for (QVector<QColor>::iterator i = colors.begin(); i!= colors.end(); i++)
        *(p++) = (::pow(BASE, i->alphaF()) - 1.0f) * scale;

    float *pointer = (float *)transferFunctionData.data();
    float stepSize = (float)width / NUMBER_TF_ENTRIES;
    float x = stepSize * 0.5f;
    int size = transferFunctionSize.at(0);
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

    std::list<QCLProgram>::iterator pVolumeRendering = QCLProgram::find(clPrograms, "Volume Rendering");
    std::list<QCLMemory>::iterator mTransferFunction = QCLMemory::find(pVolumeRendering->memories, "Transfer Function");
    memcpy((void*)mTransferFunction->getBuffer(), transferFunctionData.data(), transferFunctionData.size());
    return mTransferFunction->read(clQueue);
}

unsigned char QVRWidget::slotUpdateStepSize()
{
    if (!this->initialized) return CL_FALSE;

    int number = panel->getUI()->horizontalSliderStepSize->value();
    if (number != settings->currentStep)
    {
        settings->volumeStepSize = 1.0f / number;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QVRWidget::slotUpdateVolumeOffset()
{
    if (!this->initialized) return CL_FALSE;

    cl_float volumeOffset = panel->getUI()->horizontalSliderVolumeOffset->value() * 0.01f;
    if (volumeOffset != settings->volumeOffset)
    {
        settings->volumeOffset = volumeOffset;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QVRWidget::slotUpdateVolumeScale()
{
    if (!this->initialized) return CL_FALSE;

    cl_float volumeScale = panel->getUI()->horizontalSliderVolumeScale->value() * 0.01f;
    if (volumeScale != settings->volumeScale)
    {
        settings->volumeScale = volumeScale;
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QVRWidget::slotUpdateTimeStep()
{
    if (!this->initialized) return CL_FALSE;

    cl_uint currentStep = panel->getUI()->horizontalSliderTimeStep->value();
    if (currentStep != settings->currentStep)
    {
        std::cerr << " > LOG: QVRWidget::slotUpdateTimeStep() - " << currentStep << std::endl;
        statusMutex->lock();
        int size = cacheSize, clSize = clCacheSize;
        std::vector<cl_uint> mapping(cacheMapping), clMapping(clCacheMapping);
        std::vector<StageState> status(cacheStatus);
        for (int i = 0; i < clCacheSize; i++)
        {
            int index = i + currentStep - settings->currentStep;
            int position = (index % clSize + clSize) % clSize;
            clCacheMapping.at(i) = clMapping.at(position);
        }
        for (int i = 0; i < cacheSize; i++)
        {
            int index = i + currentStep - settings->currentStep;
            int position = (index % size + size) % size;
            cacheMapping.at(i) = mapping.at(position);
            if (index < 0 || index >= size)
                cacheStatus.at(i) = QCL_INITIALIZED;
            else if (index >= clSize)
                cacheStatus.at(i) = status.at(position) >= QCL_PREPROCESSED ? QCL_PREPROCESSED : status.at(position);
            else
                cacheStatus.at(i) = status.at(position);
        }
        settings->currentStep = currentStep;
        readingFinished->wakeAll();
        writingFinished->wakeAll();
        statusMutex->unlock();
        
        updateVolume();
        updateGL();
    }
    return GL_TRUE;
}

unsigned char QVRWidget::updateVolume()
{
    float ratio = timeSteps == 1 ? 1.0 : 1.0 / (timeSteps - 1);
    float stop0 = timeSteps == 1 || cacheSize == 1 ? 0.0 : settings->currentStep * ratio;
    float stop1 = timeSteps == 1 || clCacheSize == 1 ? 1.0 : (min(settings->currentStep + clCacheSize, timeSteps) - 1) * ratio;
    float stop2 = timeSteps == 1 || cacheSize == 1 ? 1.0 : (min(settings->currentStep + cacheSize, timeSteps) - 1) * ratio;
    std::stringstream styleSheet(std::stringstream::in | std::stringstream::out);
    styleSheet << "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:";
    styleSheet << stop0;
    styleSheet << " rgba(255, 255, 255, 255), stop:";
    styleSheet << stop0 + EPSILON;
    styleSheet << " rgba(128, 128, 128, 255), stop:";
    styleSheet << stop1;
    styleSheet << " rgba(128, 128, 128, 255), stop:";
    styleSheet << stop1 + EPSILON;
    styleSheet << " rgba(192, 192, 192, 255), stop:";
    styleSheet << stop2;
    styleSheet << " rgba(192, 192, 192, 255), stop:";
    styleSheet << stop2 + EPSILON;
    styleSheet << " rgba(255, 255, 255, 255))";
    panel->getUI()->horizontalSliderTimeStep->setStyleSheet(QString::fromLocal8Bit(styleSheet.str().c_str()));

    statusMutex->lock();
    cl_uint index = 0;
    cl_uint map = cacheMapping.at(index);
    cl_uint clMap = clCacheMapping.at(index);
    if (cacheStatus.at(index) != QCL_WRITTEN) paintingFinished->wait(statusMutex);
    statusMutex->unlock();

    emit(signalHistogramUpdated(0, (float*)cacheHistogramData.data() + map * cacheHistogramSize));

    volumeOrigin = clMap * cacheVolumeSize;
    
    return GL_TRUE;
}

void QVRWidget::initializeGL()
{
    initContext();
}

// resizeGL
void QVRWidget::resizeGL(int w, int h)
{
    if (error) return;

    settings->width = w == 0 ? 1 : w;
    settings->height = h == 0 ? 1 : h;

    updatePixelBuffer();
}

unsigned char QVRWidget::updatePixelBuffer()
{
    cl_int status = CL_SUCCESS;

    std::list<QCLProgram>::iterator pVolumeRendering = QCLProgram::find(clPrograms, "Volume Rendering");
    std::list<QCLMemory>::iterator mPixelBuffer = QCLMemory::find(pVolumeRendering->memories, "Pixel Buffer");
    if (!mPixelBuffer->initialize(clContext, std::vector<::size_t>(1, settings->width * settings->height * sizeof(GLubyte) * 4))) return GL_FALSE;

    // calculate new grid size
    gridSize.at(0) = QUtility::roundUp(localSize.at(0), settings->width);
    gridSize.at(1) = QUtility::roundUp(localSize.at(1), settings->height);

    std::list<QCLKernel>::iterator kVolumeRender = QCLKernel::find(pVolumeRendering->kernels, "Volume Render");
    status |= clSetKernelArg(kVolumeRender->get(), 1, sizeof(cl_mem), &mPixelBuffer->get());
    status |= clSetKernelArg(kVolumeRender->get(), 7, sizeof(unsigned int), &settings->width);
    status |= clSetKernelArg(kVolumeRender->get(), 8, sizeof(unsigned int), &settings->height);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    return GL_TRUE;
}

// paintGL
void QVRWidget::paintGL()
{
    if (error) return;

    glViewport(0, 0,  settings->width, settings->height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();
    glLoadIdentity();
    QVector4 rotation;
    QVector4::getAngleAxis(settings->viewRotation, rotation);
    glRotatef(-rotation.w * 180.0 / PI, rotation.x, rotation.y, rotation.z);
    glTranslatef(-settings->viewTranslation.x, -settings->viewTranslation.y, -settings->viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *)inverseViewMatrix.data());
    glPopMatrix();

    QDateTime start = QDateTime::currentDateTime();

    if (error = !drawVolume()) return;

    if (settings->enablePrintingFPS) QUtility::printFPS(start.msecsTo(QDateTime::currentDateTime()));
}

unsigned char QVRWidget::drawVolume()
{
    cl_int status = CL_SUCCESS;

    glFlush();

    // Transfer ownership of buffer from GL to CL
    std::list<QCLProgram>::iterator pVolumeRendering = QCLProgram::find(clPrograms, "Volume Rendering");
    for (std::list<QCLMemory>::iterator i = pVolumeRendering->memories.begin(); i != pVolumeRendering->memories.end(); i++)
        if (i->alwaysRead == CL_TRUE && (i->type == QCLMemory::QCL_BUFFER || i->type == QCLMemory::QCL_BUFFERGL))
            if (!i->read(clQueue)) return GL_FALSE;

    // Execute OpenCL kernel, writing results to PBO
    std::list<QCLKernel>::iterator kVolumeRender = QCLKernel::find(pVolumeRendering->kernels, "Volume Render");
    status |= clSetKernelArg(kVolumeRender->get(), 3, sizeof(cl_uint), &volumeOrigin);
    status |= clSetKernelArg(kVolumeRender->get(), 9, sizeof(cl_float), &settings->volumeOffset);
    status |= clSetKernelArg(kVolumeRender->get(), 10, sizeof(cl_float), &settings->volumeScale);
    status |= clSetKernelArg(kVolumeRender->get(), 11, sizeof(cl_float), &settings->volumeStepSize);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status = clEnqueueNDRangeKernel(clQueue, kVolumeRender->get(), gridSize.size(), NULL, gridSize.data(), localSize.data(), 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

    // Transfer ownership of buffer back from CL to GL
    for (std::list<QCLMemory>::iterator i = pVolumeRendering->memories.begin(); i != pVolumeRendering->memories.end(); i++)
        if (i->alwaysWrite == CL_TRUE && (i->type == QCLMemory::QCL_BUFFER || i->type == QCLMemory::QCL_BUFFERGL))
            if (!i->write(clQueue)) return GL_FALSE;

#ifdef __CL_ENABLE_DEBUG
    std::list<QCLMemory>::iterator mDebugOutput = QCLMemory::find(pVolumeRendering->memories, "Debug Output");
    float* ptr = (float*)mDebugOutput->getBuffer();
    std::cerr << " > LOG: " << std::endl;
#endif

    //status = clFinish(clQueue);
    //if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFinish()")) return GL_FALSE;

    // Draw image from PBO
    glRasterPos2i(0, 0);
    std::list<QCLMemory>::iterator mPixelBuffer = QCLMemory::find(pVolumeRendering->memories, "Pixel Buffer");
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *(GLuint *)mPixelBuffer->getBuffer());
    glDrawPixels(settings->width, settings->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return GL_TRUE;
}

void QVRWidget::mousePressEvent(QMouseEvent *event)
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

void QVRWidget::mouseMoveEvent(QMouseEvent *event)
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

void QVRWidget::wheelEvent(QWheelEvent *event)
{
    updateGL();
}

// keyPressEvent
void QVRWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Escape:
    case Qt::Key_Q:
        // close();
        break;
    case Qt::Key_Plus:
        settings->volumeStepSize += STEPSIZE_DELTA;
        if (settings->volumeStepSize > STEPSIZE_MAX) settings->volumeStepSize = STEPSIZE_MAX;
        break;
    case Qt::Key_Minus:
        settings->volumeStepSize -= STEPSIZE_DELTA;
        if (settings->volumeStepSize < STEPSIZE_MIN) settings->volumeStepSize = STEPSIZE_MIN;
        break;
    case Qt::Key_Comma:
        settings->volumeStepSize *= 2.0;
        if (settings->volumeStepSize > STEPSIZE_MAX) settings->volumeStepSize = STEPSIZE_MAX;
        break;
    case Qt::Key_Period:
        settings->volumeStepSize *= 0.5;
        if (settings->volumeStepSize < STEPSIZE_MIN) settings->volumeStepSize = STEPSIZE_MIN;
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

void QVRWidget::saveSettings()
{}

void QVRWidget::loadSettings()
{}

void QVRWidget::printSettings()
{
    std::cerr << " > LOG: step size " << settings->volumeStepSize << "." << std::endl;
    std::cerr << " > LOG: volume offset " << settings->volumeOffset << "." << std::endl;
    std::cerr << " > LOG: volume scale " << settings->volumeScale << "." << std::endl;
    std::cerr << " > LOG: print frames per second " << settings->enablePrintingFPS << "." << std::endl;
    std::cerr << " > LOG: light direction (" << settings->lightDirection.x << ", " << settings->lightDirection.y << ", " << settings->lightDirection.z << std::endl;
    std::cerr << " > LOG: view rotation (" << settings->viewRotation.x << ", " << settings->viewRotation.y << ", " << settings->viewRotation.z << ", " << settings->viewRotation.w << ")." << std::endl;
    std::cerr << " > LOG: view translation (" << settings->viewTranslation.x << ", " << settings->viewTranslation.y << ", " << settings->viewTranslation.z << ")." << std::endl;
    std::cerr << " > LOG: window size (" << settings->width << ", " << settings->height << ")." << std::endl;
}
