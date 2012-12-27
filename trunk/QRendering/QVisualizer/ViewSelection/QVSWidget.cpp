/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVSWidget.cpp
 * @brief   QVSWidget class declaration.
 * 
 * This file declares the methods of the main process of TVDVF defined in QVSWidget.h.
 *  
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include <gl/glew.h>
#include <cl/cl_gl.h>

#include <cmath>
#include <iomanip> 
#include <iostream>
#include <sstream>
#include <fstream>

#include <QKeyEvent>
#include <QMutex>
#include <QWaitCondition>

#include "../utilities/QIO.h"
#include "../infrastructures/QHoverPoints.h"
#include "../infrastructures/QCLProgram.h"
#include "../infrastructures/QVTKModel.h"
#include "../infrastructures/QSerializer.h"
#include "QVSControlPanel.h"
#include "QVSWidget.h"

// [houtao]
#include "float.h"

QVSWidget::QVSWidget(QWidget *parent)
    : QGLWidget(parent),
    dataFileName(), dataFilePath(), objectFileName(), // Data file
    endian(ENDIAN_LITTLE), volumeSize(), thickness(), boxSize(), timeInterval(0), format(DATA_UNKNOWN),
        volumeMin(FLT_MAX), volumeMax(-FLT_MAX), histogramMin(FLT_MAX), histogramMax(0.0f), // Volumetric Data
    transferFunctionSize(1, NUMBER_TF_ENTRIES), transferFunctionData(NUMBER_TF_ENTRIES * 4 * sizeof(float)), // Transfer Function
    cacheVolumeSize(0), cacheHistogramSize(0), cacheVolumeData(0), cacheHistogramData(0), 
        cacheVisibilityData(0), cacheEntropyData(0), cacheNoteworthinessData(0), // Memory Cache
#ifdef __CL_ENABLE_DEBUG
    cacheDebugData(0),
#endif
    initialized(CL_FALSE), windowWidth(1.0f), windowLevel(0.5f), error(GL_FALSE), settings(), // Configuration
    mouseMode(MOUSE_ROTATE), mouseX(0), mouseY(0), inverseViewMatrix(16 * sizeof(float)), // OpenGL Context
    clPrograms(), clDevices(), clContext(0), clQueue(0), // OpenCL Context
    viewSize(NUMBER_VIEW_POINTS), viewEntropy(NUMBER_VIEW_POINTS * NUMBER_VIEW_POINTS * 2), // View Entropy
    panel(NULL)
{}

QVSWidget::~QVSWidget()
{
    this->destroy();
}

unsigned char QVSWidget::destroy()
{
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
unsigned char QVSWidget::initConnections(QVSControlPanel* panel)
{
    this->panel = panel;

    const Ui::QVSControlPanel* ui = panel->getUI();
    connect(this, SIGNAL(signalHistogramInitialized(::size_t, float*)), ui->widgetEditor, SLOT(slotInsertHistogram(::size_t, float*)));
    connect(this, SIGNAL(signalHistogramUpdated(unsigned int, float*)), ui->widgetEditor, SLOT(slotUpdateHistogram(unsigned int, float*)));
    connect(this, SIGNAL(signalViewEntropyUpdated(double)), ui->labelViewEntropyValue, SLOT(setNum(double)));
    connect(this, SIGNAL(signalViewEntropyUpdated(double)), ui->widgetNorthernHemisphere, SLOT(slotUpdateViewEntropy(double)));
    connect(this, SIGNAL(signalViewEntropyUpdated(double)), ui->widgetSouthernHemisphere, SLOT(slotUpdateViewEntropy(double)));
    connect(this, SIGNAL(signalNorthernViewPointMarked(int, int)), ui->widgetNorthernHemisphere, SLOT(slotMarkePoint(int, int)));
    connect(this, SIGNAL(signalSouthernViewPointMarked(int, int)), ui->widgetSouthernHemisphere, SLOT(slotMarkePoint(int, int)));
    connect(this, SIGNAL(signalNorthernViewEntropyInitialized(::size_t, float*, float, float)), ui->widgetNorthernHemisphere, SLOT(slotInitViewEntropy(::size_t, float*, float, float)));
    connect(this, SIGNAL(signalSouthernViewEntropyInitialized(::size_t, float*, float, float)), ui->widgetSouthernHemisphere, SLOT(slotInitViewEntropy(::size_t, float*, float, float)));
    connect(this, SIGNAL(signalLoadViewEntropy()), ui->widgetEditor, SLOT(slotLoadConfigurations()));
    connect(this, SIGNAL(signalSaveViewEntropy()), ui->widgetEditor, SLOT(slotSaveConfigurations()));

    connect(ui->widgetEditor, SIGNAL(signalControlPointsChanged(QHoverPoints*, int, unsigned char)), this, SLOT(slotUpdateTransferFunction(QHoverPoints*, int, unsigned char)));
    connect(ui->horizontalSliderStepSize, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateVolumeStepSize(int)));
    connect(ui->horizontalSliderVolumeOffset, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateVolumeOffset(int)));
    connect(ui->horizontalSliderVolumeScale, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateVolumeScale(int)));
    connect(ui->horizontalSliderMaterialShininess, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateMaterialShininess(int)));
    connect(ui->doubleSpinBoxLightPositionX, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightPositionX(double)));
    connect(ui->doubleSpinBoxLightPositionY, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightPositionY(double)));
    connect(ui->doubleSpinBoxLightPositionZ, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightPositionZ(double)));
    connect(ui->doubleSpinBoxLightDiffuse, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightDiffuseCoeff(double)));
    connect(ui->doubleSpinBoxLightAmbient, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightAmbientCoeff(double)));
    connect(ui->doubleSpinBoxLightSpecular, SIGNAL(valueChanged(double)), this, SLOT(slotUpdateLightSpecularCoeff(double)));
    connect(ui->groupBoxVisibility, SIGNAL(toggled(bool)), this, SLOT(slotUpdateComputingEntropyState(bool)));
    connect(ui->groupBoxIlluminationModel, SIGNAL(toggled(bool)), this, SLOT(slotUpdateShadingState(bool)));
    connect(ui->radioButtonGaussian1D1, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian1DState(bool)));
    connect(ui->radioButtonGaussian1D3, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian1DState(bool)));
    connect(ui->radioButtonGaussian1D5, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian1DState(bool)));
    connect(ui->radioButtonGaussian1D7, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian1DState(bool)));
    connect(ui->radioButtonGaussian2D1, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian2DState(bool)));
    connect(ui->radioButtonGaussian2D3, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian2DState(bool)));
    connect(ui->radioButtonGaussian2D5, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian2DState(bool)));
    connect(ui->radioButtonGaussian2D7, SIGNAL(toggled(bool)), this, SLOT(slotUpdateGaussian2DState(bool)));
    connect(ui->pushButtonCompute, SIGNAL(clicked()), this, SLOT(slotComputeViewEntropy()));
    connect(ui->pushButtonStart, SIGNAL(clicked()), this, SLOT(slotMarkStartPoint()));
    connect(ui->pushButtonEnd, SIGNAL(clicked()), this, SLOT(slotMarkEndPoint()));
    connect(ui->pushButtonLoad, SIGNAL(clicked()), this, SLOT(slotLoadConfigurations()));
    connect(ui->pushButtonSave, SIGNAL(clicked()), this, SLOT(slotSaveConfigurations()));

    connect(panel->colorDialog, SIGNAL(colorSelected(const QColor&)), this, SLOT(slotUpdateLightColor(const QColor&)));

    settings.volumeStepSize = 1.0f / ui->horizontalSliderStepSize->value();
    settings.volumeOffset = ui->horizontalSliderVolumeOffset->value() * 0.01f;
    settings.volumeScale = ui->horizontalSliderVolumeScale->value() * 0.01f;
    settings.enableComputingEntropy = ui->groupBoxVisibility->isChecked();
    settings.enableShading = ui->groupBoxIlluminationModel->isChecked();
    settings.materialShininess = ui->horizontalSliderMaterialShininess->value();
    settings.lightDirection.x = ui->doubleSpinBoxLightPositionX->value();
    settings.lightDirection.y = ui->doubleSpinBoxLightPositionY->value();
    settings.lightDirection.z = ui->doubleSpinBoxLightPositionZ->value();
    settings.diffuseCoeff = ui->doubleSpinBoxLightDiffuse->value();
    settings.ambientCoeff = ui->doubleSpinBoxLightAmbient->value();
    settings.specularCoeff = ui->doubleSpinBoxLightSpecular->value();

    if (panel->getUI()->radioButtonGaussian1D7->isChecked())
        settings.gaussian1D = 7;
    else if (panel->getUI()->radioButtonGaussian1D5->isChecked())
        settings.gaussian1D = 5;
    else if (panel->getUI()->radioButtonGaussian1D3->isChecked())
        settings.gaussian1D = 3;
    else
        settings.gaussian1D = 1;

    if (panel->getUI()->radioButtonGaussian2D7->isChecked())
        settings.gaussian2D = 7;
    else if (panel->getUI()->radioButtonGaussian2D5->isChecked())
        settings.gaussian2D = 5;
    else if (panel->getUI()->radioButtonGaussian2D3->isChecked())
        settings.gaussian2D = 3;
    else
        settings.gaussian2D = 1;

    return GL_TRUE;
}

// Step 2 - init data
unsigned char QVSWidget::initData(const std::string &name)
{
    dataFileName = name;
    int position = dataFileName.find_last_of("\\");
    if (position == std::string::npos) position = dataFileName.find_last_of("/");
    if (position == std::string::npos) position = dataFileName.size() - 1;
    dataFilePath = dataFileName.substr(0, position + 1);

    if (error = !parseDataFile(dataFileName)) return GL_FALSE;
    
    return GL_TRUE;
}

unsigned char QVSWidget::parseDataFile(const std::string &name)
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
void QVSWidget::initContext()
{
    if (error |= !QUtility::checkSupport()) return;
    if (error |= !initOpenCL()) return;
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    this->initialized = true;
}

unsigned char QVSWidget::initOpenCL()
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

unsigned char QVSWidget::initConfigurations()
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

    settings.volumeScale = windowWidth < EPSILON ? 1.0f : 1.0f / windowWidth;
    settings.volumeOffset = windowWidth * 0.5f - windowLevel;

    cacheVolumeData.resize(CACHE_VOLUME_SIZE);
    memset(cacheVolumeData.data(), 0, cacheVolumeData.size());
    if (error = cacheVolumeData.size() != CACHE_VOLUME_SIZE)
    {
        std::cerr << " > ERROR: allocating volume memory(" << CACHE_VOLUME_SIZE << "B) failed." << std::endl;
        return GL_FALSE;
    }
    cacheVolumeSize = volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2];
    if (error = cacheVolumeSize * 4 * sizeof(float) > cacheVolumeData.size())
    {
        std::cerr << " > ERROR: limited volume memory allocated." << std::endl;
        return GL_FALSE;
    }

    cacheHistogramSize = NUMBER_HIS_ENTRIES;
    ::size_t size = cacheHistogramSize * sizeof(float);
    cacheHistogramData.resize(size);
    memset(cacheHistogramData.data(), 0, cacheHistogramData.size());
    if (error = cacheHistogramData.size() != size)
    {
        std::cerr << " > ERROR: allocating histogram memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }

    size = cacheHistogramSize * sizeof(float);
    cacheVisibilityData.resize(size);
    memset(cacheVisibilityData.data(), 0, cacheVisibilityData.size());
    if (error = cacheVisibilityData.size() != size)
    {
        std::cerr << " > ERROR: allocating visibility memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }

    cacheEntropyData.resize(size);
    memset(cacheEntropyData.data(), 0, cacheEntropyData.size());
    if (error = cacheEntropyData.size() != size)
    {
        std::cerr << " > ERROR: allocating entropy memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }

    cacheNoteworthinessData.resize(size);
    memset(cacheNoteworthinessData.data(), 0, cacheNoteworthinessData.size());
    if (error = cacheNoteworthinessData.size() != size)
    {
        std::cerr << " > ERROR: allocating noteworthiness memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }

#ifdef __CL_ENABLE_DEBUG
    size = CACHE_CL_DEBUG_SIZE;
    cacheDebugSize = size / 4;
    cacheDebugData.resize(size);
    memset(cacheDebugData.data(), 0, cacheDebugData.size());
    if (error = cacheDebugData.size() != size)
    {
        std::cerr << " > ERROR: allocating debug memory(" << size << "B) failed." << std::endl;
        return GL_FALSE;
    }
#endif
    return GL_TRUE;
}

unsigned char QVSWidget::initPrograms()
{
    cl_int status = CL_SUCCESS;
    
    clPrograms.push_front(QCLProgram("View Selection", "./cl/ViewSelection/"));
    std::list<QCLProgram>::iterator pViewSelection = clPrograms.begin();
    pViewSelection->kernels.push_front(QCLKernel(pViewSelection->path, "Volume Render", "volumeRendering", "view_selector.cl"));
    std::list<QCLKernel>::iterator kVolumeRender = pViewSelection->kernels.begin();
    pViewSelection->kernels.push_front(QCLKernel(pViewSelection->path, "Volume Gradient", "volumeGradient", "view_gradient.cl"));
    std::list<QCLKernel>::iterator kVolumeGradient = pViewSelection->kernels.begin();
    pViewSelection->kernels.push_front(QCLKernel(pViewSelection->path, "Volume Smoother", "volumeSmoothing", "view_smoother.cl"));
    std::list<QCLKernel>::iterator kVolumeSmoother = pViewSelection->kernels.begin();
    pViewSelection->kernels.push_front(QCLKernel(pViewSelection->path, "View Init", "viewInit", "view_init.cl"));
    std::list<QCLKernel>::iterator kViewInit = pViewSelection->kernels.begin();
    pViewSelection->kernels.push_front(QCLKernel(pViewSelection->path, "View Final", "viewFinal", "view_final.cl"));
    std::list<QCLKernel>::iterator kViewFinal = pViewSelection->kernels.begin();
    if (!pViewSelection->initialize(clContext, clDevices, getOptions())) return GL_FALSE;

    pViewSelection->memories.push_front(QCLMemory("Inverse View Matrix", QCLMemory::QCL_BUFFER, CL_TRUE, CL_FALSE, CL_MEM_READ_ONLY));
    std::list<QCLMemory>::iterator mInverseViewMatrix = pViewSelection->memories.begin();
    if (!mInverseViewMatrix->initialize(clContext, std::vector<::size_t>(1, inverseViewMatrix.size()), inverseViewMatrix.data())) return GL_FALSE;
    status = clSetKernelArg(kVolumeRender->get(), 0, sizeof(cl_mem), &mInverseViewMatrix->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    pViewSelection->memories.push_front(QCLMemory("Color Buffer", QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_WRITE));
    std::list<QCLMemory>::iterator mColorBuffer = pViewSelection->memories.begin();
    if (!mColorBuffer->initialize(clContext, std::vector<::size_t>(1, settings.width * settings.height * sizeof(GLfloat) * 4))) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 1, sizeof(cl_mem), &mColorBuffer->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    cl_uint4 size = volumeSize;
    size.s[3] = sizeof(cl_float4);
    pViewSelection->memories.push_front(QCLMemory("Volume Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_WRITE, size));
    std::list<QCLMemory>::iterator mVolumeData = pViewSelection->memories.begin();
    if (!mVolumeData->initialize(clContext, std::vector<::size_t>(1, mVolumeData->getSize()), cacheVolumeData.data())) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 2, sizeof(cl_mem), &mVolumeData->get());
    status |= clSetKernelArg(kVolumeRender->get(), 3, sizeof(cl_uint4), &volumeSize);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
    
    cl_image_format fTransferFunction;
    fTransferFunction.image_channel_order = CL_RGBA;
    fTransferFunction.image_channel_data_type = CL_FLOAT;
    pViewSelection->memories.push_front(QCLMemory("Transfer Function", QCLMemory::QCL_IMAGE2D, CL_FALSE, CL_FALSE, CL_MEM_READ_ONLY, fTransferFunction));
    std::list<QCLMemory>::iterator mTransferFunction = pViewSelection->memories.begin();
    if (!mTransferFunction->initialize(clContext, std::vector<::size_t>(settings.transferFunctionSize.begin(), settings.transferFunctionSize.end()), transferFunctionData.data())) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 4, sizeof(cl_mem), &mTransferFunction->get());
    status |= clSetKernelArg(kVolumeRender->get(), 5, sizeof(cl_sampler), &mTransferFunction->getSampler());
    status |= clSetKernelArg(kVolumeRender->get(), 11, sizeof(cl_float4), &boxSize);
    status |= clSetKernelArg(kVolumeRender->get(), 12, sizeof(cl_uint), &cacheHistogramSize);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    pViewSelection->memories.push_front(QCLMemory("CL Buffer Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_WRITE));
    std::list<QCLMemory>::iterator mCLBufferData = pViewSelection->memories.begin();
    if (!mCLBufferData->initialize(clContext, std::vector<size_t>(1, CACHE_CL_BUFFER_SIZE))) return GL_FALSE;
    status |= clSetKernelArg(kVolumeRender->get(), 18, sizeof(cl_mem), &mCLBufferData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
    
#ifdef __CL_ENABLE_DEBUG
    pViewSelection->memories.push_front(QCLMemory("Debug Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_TRUE, CL_MEM_WRITE_ONLY));
    std::list<QCLMemory>::iterator mDebugData = pViewSelection->memories.begin();
    if (!mDebugData->initialize(clContext, std::vector<::size_t>(1, CACHE_CL_DEBUG_SIZE), cacheDebugData.data())) return GL_FALSE;
    status = clSetKernelArg(kVolumeRender->get(), 19, sizeof(cl_uint), &cacheDebugSize);
    status = clSetKernelArg(kVolumeRender->get(), 20, sizeof(cl_mem), &mDebugData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
#endif

    status |= clSetKernelArg(kVolumeGradient->get(), 0, sizeof(cl_mem), &mVolumeData->get());
    status |= clSetKernelArg(kVolumeGradient->get(), 1, sizeof(cl_uint4), &volumeSize);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    pViewSelection->memories.push_front(QCLMemory("Pixel Buffer", QCLMemory::QCL_BUFFERGL, CL_TRUE, CL_TRUE, CL_MEM_WRITE_ONLY));
    std::list<QCLMemory>::iterator mPixelBuffer = pViewSelection->memories.begin();
    if (!mPixelBuffer->initialize(clContext, std::vector<::size_t>(1, settings.width * settings.height * sizeof(GLubyte) * 4))) return GL_FALSE;
    status |= clSetKernelArg(kVolumeSmoother->get(), 0, sizeof(cl_mem), &mColorBuffer->get());
    status |= clSetKernelArg(kVolumeSmoother->get(), 1, sizeof(cl_mem), &mPixelBuffer->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status |= clSetKernelArg(kViewInit->get(), 0, sizeof(cl_uint), &cacheHistogramSize);
    status |= clSetKernelArg(kViewInit->get(), 1, sizeof(cl_mem), &mCLBufferData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status |= clSetKernelArg(kViewFinal->get(), 1, sizeof(cl_uint), &cacheHistogramSize);
    status |= clSetKernelArg(kViewFinal->get(), 2, sizeof(cl_mem), &mCLBufferData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
    
    size.s[0] = cacheHistogramSize;
    size.s[1] = size.s[2] = 1;
    size.s[3] = sizeof(cl_float);
    GLboolean enabled = panel->getUI()->groupBoxVisibility->isChecked();
    pViewSelection->memories.push_front(QCLMemory("Visibility Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_TRUE, CL_MEM_READ_WRITE, size));
    std::list<QCLMemory>::iterator mVisibilityData = pViewSelection->memories.begin();
    mVisibilityData->enabled = enabled;
    if (!mVisibilityData->initialize(clContext, std::vector<::size_t>(1, mVisibilityData->getSize()), cacheVisibilityData.data())) return GL_FALSE;
    status = clSetKernelArg(kViewFinal->get(), 3, sizeof(cl_mem), &mVisibilityData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    pViewSelection->memories.push_front(QCLMemory("Entropy Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_TRUE, CL_MEM_READ_WRITE, size));
    std::list<QCLMemory>::iterator mEntropyData = pViewSelection->memories.begin();
    mEntropyData->enabled = enabled;
    if (!mEntropyData->initialize(clContext, std::vector<::size_t>(1, mEntropyData->getSize()), cacheEntropyData.data())) return GL_FALSE;
    status = clSetKernelArg(kViewFinal->get(), 4, sizeof(cl_mem), &mEntropyData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
    
    updatePixelBuffer();

    return GL_TRUE;
}

unsigned char QVSWidget::initArguments()
{
    cl_int status = CL_SUCCESS;

    // load volume data
    float volumeMin = FLT_MAX, volumeMax = FLT_MIN;
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
    QUtility::preprocess(cacheVolumeData.data(), cacheVolumeSize, format, endian, cacheHistogramSize, (float*)cacheHistogramData.data(),
        volumeMin, volumeMax, histogramMin, histogramMax);

    // format volume data
    float* begin = (float*)cacheVolumeData.data();
    float* end = begin + cacheVolumeSize;
    float* destination = begin + cacheVolumeSize * 4;
    for (float *source = end; source != begin; )
    {
        *(--destination) = *(--source);
        *(--destination) = 0.0f;
        *(--destination) = 0.0f;
        *(--destination) = 0.0f;
    }

    std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mVolumeData = QCLMemory::find(pMarchingCubes->memories, "Volume Data");
    mVolumeData->read(clQueue);
    
    // initialize UI parameters
    emit signalHistogramInitialized(cacheHistogramSize, (float*)cacheHistogramData.data());
    emit signalHistogramInitialized(cacheHistogramSize, (float*)cacheEntropyData.data());

    emit signalNorthernViewEntropyInitialized(viewSize, viewEntropy.data(), FLT_MAX, 0.0f);
    emit signalSouthernViewEntropyInitialized(viewSize, viewEntropy.data() + viewSize * viewSize, FLT_MAX, 0.0f);

    QTransferFunction1D *widgetEditor = panel->getUI()->widgetEditor;
    updateTransferFunction(widgetEditor->getHoverPoints(), widgetEditor->width());

    // compute volume gradient
    QDateTime start = QDateTime::currentDateTime();
    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLKernel>::iterator kVolumeGradient = QCLKernel::find(pViewSelection->kernels, "Volume Gradient");
    std::vector<::size_t> lSize(2, 16);
    std::vector<::size_t> gSize(2);
    gSize.at(0) = QUtility::roundUp(lSize.at(0), volumeSize.s[0]);
    gSize.at(1) = QUtility::roundUp(lSize.at(1), volumeSize.s[1]);
    for (cl_uint i = 0; i < volumeSize.s[2]; i++)
    {
        status = clSetKernelArg(kVolumeGradient->get(), 2, sizeof(cl_uint), &i);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

        status = clEnqueueNDRangeKernel(clQueue, kVolumeGradient->get(), gSize.size(), NULL, gSize.data(), lSize.data(), 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

        status = clFlush(clQueue);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFlush()")) return GL_FALSE;
    }
    status = clFinish(clQueue);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFinish()")) return GL_FALSE;
    if (settings.enablePrintingFPS) QUtility::printFPS(start.msecsTo(QDateTime::currentDateTime()), "volumeGradient()");

    return GL_TRUE;
}

// Step 4 - message loop

// public slots
void QVSWidget::slotUpdateTransferFunction(QHoverPoints *controlPoints, int width, unsigned char modified)
{
    if (modified)
    {
        memset(viewEntropy.data(), 0, viewEntropy.size() * sizeof(cl_float));
        emit signalNorthernViewEntropyInitialized(viewSize, viewEntropy.data(), FLT_MAX, 0.0f);
        emit signalSouthernViewEntropyInitialized(viewSize, viewEntropy.data() + viewSize * viewSize, FLT_MAX, 0.0f);
    }

    updateTransferFunction(controlPoints, width);
    updateGL();
}

unsigned char QVSWidget::updateTransferFunction(QHoverPoints *controlPoints, int width)
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

    if (!this->initialized) return CL_FALSE;
    
    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mTransferFunction = QCLMemory::find(pViewSelection->memories, "Transfer Function");
    memcpy((void*)mTransferFunction->getBuffer(), transferFunctionData.data(), transferFunctionData.size());
    if (!mTransferFunction->read(clQueue)) return CL_FALSE;

    return CL_TRUE;
}

void QVSWidget::slotUpdateVolumeStepSize(int value)
{
    settings.volumeStepSize = 1.0f / panel->getUI()->horizontalSliderStepSize->value();
    updateGL();
}

void QVSWidget::slotUpdateVolumeOffset(int value)
{
    settings.volumeOffset = panel->getUI()->horizontalSliderVolumeOffset->value() * 0.01f;
    updateGL();
}

void QVSWidget::slotUpdateVolumeScale(int value)
{
    settings.volumeScale = panel->getUI()->horizontalSliderVolumeScale->value() * 0.01f;
    updateGL();
}

void QVSWidget::slotUpdateLightPositionX(double value)
{
    settings.lightDirection.x = value;
    updateGL();
}

void QVSWidget::slotUpdateLightPositionY(double value)
{
    settings.lightDirection.y = value;
    updateGL();
}

void QVSWidget::slotUpdateLightPositionZ(double value)
{
    settings.lightDirection.z = value;
    updateGL();
}

void QVSWidget::slotUpdateLightColor(const QColor &color)
{
    switch (panel->currentPushButton)
    {
    case 0:
        settings.lightAmbient.x = color.redF();
        settings.lightAmbient.y = color.greenF();
        settings.lightAmbient.z = color.greenF();
    	break;
    case 1:
        settings.lightDiffuse.x = color.redF();
        settings.lightDiffuse.y = color.greenF();
        settings.lightDiffuse.z = color.greenF();
        break;
    case 2:
        settings.lightSpecular.x = color.redF();
        settings.lightSpecular.y = color.greenF();
        settings.lightSpecular.z = color.greenF();
        break;
    }
    updateGL();
}

void QVSWidget::slotUpdateLightDiffuseCoeff(double value)
{
    settings.diffuseCoeff = value;
    updateGL();
}

void QVSWidget::slotUpdateLightAmbientCoeff(double value)
{
    settings.ambientCoeff = value;
    updateGL();
}

void QVSWidget::slotUpdateLightSpecularCoeff(double value)
{
    settings.specularCoeff = value;
    updateGL();
}

void QVSWidget::slotUpdateMaterialShininess(int value)
{
    settings.materialShininess = value;
    updateGL();
}

void QVSWidget::slotUpdateComputingEntropyState(bool value)
{
    GLboolean enabled = value ? 1 : 0;
    settings.enableComputingEntropy = enabled;

    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mVisibilityData = QCLMemory::find(pViewSelection->memories, "Visibility Data");
    std::list<QCLMemory>::iterator mEntropyData = QCLMemory::find(pViewSelection->memories, "Entropy Data");
    mVisibilityData->enabled = mEntropyData->enabled = enabled;

    if (!enabled)
    {
        for (int i = 0; i < 3; i++)
        {
            emit signalNorthernViewPointMarked(i, -1);
            emit signalSouthernViewPointMarked(i, -1);
        }
        emit signalViewEntropyUpdated(0.0);
        
        memset(cacheEntropyData.data(), 0, cacheEntropyData.size());
        memset(cacheVisibilityData.data(), 0, cacheVisibilityData.size());
    }

    clPrograms.clear();
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    updateGL();
}

void QVSWidget::slotUpdateShadingState(bool value)
{
    GLboolean enabled = value ? 1 : 0;
    settings.enableShading = enabled;

    clPrograms.clear();
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    updateGL();
}

void QVSWidget::slotUpdateGaussian1DState(bool value)
{
    if (panel->getUI()->radioButtonGaussian1D7->isChecked())
        settings.gaussian1D = 7;
    else if (panel->getUI()->radioButtonGaussian1D5->isChecked())
        settings.gaussian1D = 5;
    else if (panel->getUI()->radioButtonGaussian1D3->isChecked())
        settings.gaussian1D = 3;
    else
        settings.gaussian1D = 1;

    clPrograms.clear();
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    updateGL();
}

void QVSWidget::slotUpdateGaussian2DState(bool value)
{
    if (panel->getUI()->radioButtonGaussian2D7->isChecked())
        settings.gaussian2D = 7;
    else if (panel->getUI()->radioButtonGaussian2D5->isChecked())
        settings.gaussian2D = 5;
    else if (panel->getUI()->radioButtonGaussian2D3->isChecked())
        settings.gaussian2D = 3;
    else
        settings.gaussian2D = 1;

    clPrograms.clear();
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    updateGL();
}

void QVSWidget::initializeGL()
{
    initContext();
}

// resizeGL
void QVSWidget::resizeGL(int w, int h)
{
    if (error) return;

    settings.width = w == 0 ? 1 : w;
    settings.height = h == 0 ? 1 : h;

    updatePixelBuffer();
}

unsigned char QVSWidget::updatePixelBuffer()
{
    cl_int status = CL_SUCCESS;

    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mPixelBuffer = QCLMemory::find(pViewSelection->memories, "Pixel Buffer");
    std::list<QCLMemory>::iterator mColorBuffer = QCLMemory::find(pViewSelection->memories, "Color Buffer");
    ::size_t size(settings.width * settings.height * sizeof(GLubyte) * 4);
    if (size > mPixelBuffer->size.at(0))
    {
        size += size / 2;
        if (!mPixelBuffer->initialize(clContext, std::vector<::size_t>(1, size))) return GL_FALSE;
        if (!mColorBuffer->initialize(clContext, std::vector<::size_t>(1, size / sizeof(GLubyte) * sizeof(GLfloat)))) return GL_FALSE;
    }

    std::list<QCLKernel>::iterator kVolumeRender = QCLKernel::find(pViewSelection->kernels, "Volume Render");
    status |= clSetKernelArg(kVolumeRender->get(), 1, sizeof(cl_mem), &mColorBuffer->get());
    status |= clSetKernelArg(kVolumeRender->get(), 6, sizeof(unsigned int), &settings.width);
    status |= clSetKernelArg(kVolumeRender->get(), 7, sizeof(unsigned int), &settings.height);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    std::list<QCLKernel>::iterator kVolumeSmoother = QCLKernel::find(pViewSelection->kernels, "Volume Smoother");
    status |= clSetKernelArg(kVolumeSmoother->get(), 0, sizeof(cl_mem), &mColorBuffer->get());
    status |= clSetKernelArg(kVolumeSmoother->get(), 1, sizeof(cl_mem), &mPixelBuffer->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    return GL_TRUE;
}

// paintGL
void QVSWidget::paintGL()
{
    if (error) return;
    
    glViewport(0, 0,  settings.width, settings.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();
    glLoadIdentity();
    QVector4 angleAxis;
    QVector4::getAngleAxis(settings.viewRotation, angleAxis);
    glRotatef(-angleAxis.w * 180.0 / PI, angleAxis.x, angleAxis.y, angleAxis.z);
    glTranslatef(-settings.viewTranslation.x, -settings.viewTranslation.y, -settings.viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *)inverseViewMatrix.data());
    glPopMatrix();
    
    QDateTime start = QDateTime::currentDateTime();
    if (error = !drawVolume()) return;
    if (settings.enablePrintingFPS) QUtility::printFPS(start.msecsTo(QDateTime::currentDateTime()), "drawVolume()");
}

unsigned char QVSWidget::computeHistograms(unsigned char enableComputingEntropy)
{
    cl_int status = CL_SUCCESS;

    // calculate new grid size
    std::vector<::size_t> localSize(2, ::size_t(16));
    std::vector<::size_t> globalSize(2);
    globalSize.at(0) = QUtility::roundUp(localSize.at(0), settings.width  / 2);
    globalSize.at(1) = QUtility::roundUp(localSize.at(1), settings.height / 2);

    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");

    //
    // Step 1: initialize the cache buffer to compute the visibility of current view
    //
    if (enableComputingEntropy)
    {
        std::vector<::size_t> lSize(3, 1);
        std::vector<::size_t> gSize(3, 0);
        gSize.at(0) = globalSize.at(0) / localSize.at(0);
        gSize.at(1) = globalSize.at(1) / localSize.at(1);
        gSize.at(2) = cacheHistogramSize;
        cl_uint groupSize(gSize.at(0) * gSize.at(1));
        if (CACHE_CL_BUFFER_SIZE < groupSize * cacheHistogramSize * 2 * sizeof(cl_uint))
        {
            std::cerr << " > ERROR: not enough GPU memory for cl buffer." << std::endl;
            return GL_FALSE;
        }

        std::list<QCLKernel>::iterator kViewInit = QCLKernel::find(pViewSelection->kernels, "View Init");
        status = clEnqueueNDRangeKernel(clQueue, kViewInit->get(), gSize.size(), NULL, gSize.data(), lSize.data(), 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

        status = clFlush(clQueue);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFlush()")) return GL_FALSE;
    }
    
    //
    // Step 2: render the volume while computing the visibilities of all the voxels
    //
    // Transfer ownership of buffer from GL to CL
    for (std::list<QCLMemory>::iterator i = pViewSelection->memories.begin(); i != pViewSelection->memories.end(); i++)
        if (i->alwaysRead == CL_TRUE && (i->type == QCLMemory::QCL_BUFFER || i->type == QCLMemory::QCL_BUFFERGL))
            if (!i->read(clQueue)) return GL_FALSE;

    cl_float4 diffuse  = { settings.lightDiffuse.x  * settings.diffuseCoeff,  settings.lightDiffuse.y  * settings.diffuseCoeff,  settings.lightDiffuse.z  * settings.diffuseCoeff,  0.0f };
    cl_float4 specular = { settings.lightSpecular.x * settings.specularCoeff, settings.lightSpecular.y * settings.specularCoeff, settings.lightSpecular.z * settings.specularCoeff, 0.0f };
    cl_float4 ambient  = { settings.lightAmbient.x  * settings.ambientCoeff,  settings.lightAmbient.y  * settings.ambientCoeff,  settings.lightAmbient.z  * settings.ambientCoeff,  1.0f };

    // Execute OpenCL kernel, writing results to PBO
    std::list<QCLKernel>::iterator kVolumeRender = QCLKernel::find(pViewSelection->kernels, "Volume Render");
    status |= clSetKernelArg(kVolumeRender->get(), 8, sizeof(cl_float), &settings.volumeOffset);
    status |= clSetKernelArg(kVolumeRender->get(), 9, sizeof(cl_float), &settings.volumeScale);
    status |= clSetKernelArg(kVolumeRender->get(), 10, sizeof(cl_float), &settings.volumeStepSize);
    status |= clSetKernelArg(kVolumeRender->get(), 13, sizeof(cl_float4), &diffuse);
    status |= clSetKernelArg(kVolumeRender->get(), 14, sizeof(cl_float4), &specular);
    status |= clSetKernelArg(kVolumeRender->get(), 15, sizeof(cl_float4), &ambient);
    status |= clSetKernelArg(kVolumeRender->get(), 16, sizeof(cl_float4), &settings.lightDirection);
    status |= clSetKernelArg(kVolumeRender->get(), 17, sizeof(cl_float), &settings.materialShininess);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status = clEnqueueNDRangeKernel(clQueue, kVolumeRender->get(), globalSize.size(), NULL, globalSize.data(), localSize.data(), 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

    status = clFlush(clQueue);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFlush()")) return GL_FALSE;

    std::vector<::size_t> lSize(2, 1);
    std::vector<::size_t> gSize(2, 0);
    gSize.at(0) = settings.width;
    gSize.at(1) = settings.height;
    std::list<QCLKernel>::iterator kVolumeSmoother = QCLKernel::find(pViewSelection->kernels, "Volume Smoother");
    status = clEnqueueNDRangeKernel(clQueue, kVolumeSmoother->get(), gSize.size(), NULL, gSize.data(), lSize.data(), 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

    //
    // Step 3: compute the visibility histograms
    //
    if (enableComputingEntropy)
    {
        status = clFlush(clQueue);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFlush()")) return GL_FALSE;
        
        std::vector<::size_t> lSize(2, 1);
        std::vector<::size_t> gSize(2, 0);
        gSize.at(0) = cacheHistogramSize;
        gSize.at(1) = 2;
        std::list<QCLKernel>::iterator kViewFinal = QCLKernel::find(pViewSelection->kernels, "View Final");
        cl_uint groupSize((globalSize.at(0) / localSize.at(0)) * (globalSize.at(1) / localSize.at(1)));
        status |= clSetKernelArg(kViewFinal->get(), 0, sizeof(cl_uint), &groupSize);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

        status = clEnqueueNDRangeKernel(clQueue, kViewFinal->get(), gSize.size(), NULL, gSize.data(), lSize.data(), 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;
    }

    // Transfer ownership of buffer back from CL to GL
    for (std::list<QCLMemory>::iterator i = pViewSelection->memories.begin(); i != pViewSelection->memories.end(); i++)
        if (i->alwaysWrite == CL_TRUE && (i->type == QCLMemory::QCL_BUFFER || i->type == QCLMemory::QCL_BUFFERGL))
            if (!i->write(clQueue)) return GL_FALSE;

    status = clFinish(clQueue);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFinish()")) return GL_FALSE;

#ifdef __CL_ENABLE_DEBUG
    float* pDebug = (float*)cacheDebugData.data();
    std::cerr << " > LOG: " << std::endl;
    /*
    std::list<QCLMemory>::iterator mDebugData = QCLMemory::find(pViewSelection->memories, "Debug Data");
    float* ptr = (float*)mDebugData->getBuffer();
    std::cerr << " > LOG: " << std::endl;
    */
#endif

    return GL_TRUE;
}

unsigned char QVSWidget::computeEntropy(const QVector4 &rotation, unsigned char paint)
{
    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mVisibilityData = QCLMemory::find(pViewSelection->memories, "Visibility Data");
    std::list<QCLMemory>::iterator mEntropyData = QCLMemory::find(pViewSelection->memories, "Entropy Data");
    float* pVisibility = (cl_float*)mVisibilityData->getBuffer();
    float* pEntropy = (cl_float*)mEntropyData->getBuffer();
    float* pTransferFunction = (float*)transferFunctionData.data() + 3;
    float* pHistogram = (float*)cacheHistogramData.data();
    float* pNoteworthiness = (float*)cacheNoteworthinessData.data();
    float sigma(0.0f), logBias(::log((float)cacheVolumeSize) / ::log(2.0));
    for (int i = 0; i < cacheHistogramSize; i++)
    {
        *pNoteworthiness = *pTransferFunction * (logBias - (histogramMin + *pHistogram * (histogramMax - histogramMin)));
        if (*pNoteworthiness > 0.0f)
        {
            float t = 1.0 / *pNoteworthiness;
            sigma += t * *pVisibility;
        }
        pTransferFunction += 4;
        pVisibility++;
        pHistogram++;
        pNoteworthiness++;
    }

    pVisibility = (float*)mVisibilityData->getBuffer();
    pNoteworthiness = (float*)cacheNoteworthinessData.data();

    double entropy(0.0), logScale(1.0 / ::log(2.0));
    if (sigma > 0.0)
    {
        sigma = 1.0 / sigma;
        for (int i = 0; i < cacheHistogramSize; i++)
        {
            if (*pNoteworthiness > 0.0f)
            {
                double t = sigma / *pNoteworthiness;
                entropy += -t * ((::log(t) * logScale * *pVisibility) + *pEntropy);
            }
            pVisibility++;
            pEntropy++;
            pNoteworthiness++;
        }
    }

    int northernOffset(-1), southernOffset(-1);
    if (entropy > 0.0)
    {
        // Quaternion
        // Given the unit quaternion q = (w,x,y,z), the equivalent 3¡Á3 rotation matrix is
        // 1 - 2*y^2 - 2*z^2    & 2*x*y - 2*z*w        & 2*x*z + 2*y*w
        //     2*x*y + 2*z*w    & 1 - 2*x^2 - 2*z^2    & 2*y*z - 2*x*w
        //     2*x*z - 2*y*w    & 2*y*z + 2*x*w        & 1 - 2*x^2 - 2*y^2
        QVector3 normal(
            2.0f * (rotation.x * rotation.z + rotation.y * rotation.w),
            2.0f * (rotation.y * rotation.z - rotation.x * rotation.w),
            1.0f - 2.0f * (rotation.x * rotation.x + rotation.y * rotation.y)
            );

        float viewScale = (viewSize - 1) * 0.5;
        if (normal.z >= 0)
        {
            float x(viewScale * (-normal.x + 1.0f)), y(viewScale * (normal.y + 1.0f));
            northernOffset = (int)x + (int)y * viewSize;
            viewEntropy.at(northernOffset) = entropy;
        }
        else
        {
            float x(viewScale * (normal.x + 1.0f)), y(viewScale * (-normal.y + 1.0f));
            southernOffset = (int)x + (int)y * viewSize;
            viewEntropy.at(viewSize * viewSize + southernOffset) = entropy;
        }
    }

    if (paint)
    {
        emit signalNorthernViewPointMarked(2, northernOffset);
        emit signalSouthernViewPointMarked(2, southernOffset);
        emit signalViewEntropyUpdated(entropy);

        QUtilityTemplate<float>::computeLogarithm(cacheHistogramSize, (float*)mEntropyData->getBuffer());
        emit signalHistogramUpdated(0, NULL);
    }
    
    return GL_TRUE;
}

unsigned char QVSWidget::drawVolume()
{
    glFlush();

    computeHistograms(settings.enableComputingEntropy);
    if (settings.enableComputingEntropy) computeEntropy(settings.viewRotation);

    // Draw image from PBO
    glRasterPos2i(0, 0);
    std::list<QCLProgram>::iterator pViewSelection = QCLProgram::find(clPrograms, "View Selection");
    std::list<QCLMemory>::iterator mPixelBuffer = QCLMemory::find(pViewSelection->memories, "Pixel Buffer");
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *(GLuint *)mPixelBuffer->getBuffer());
    glDrawPixels(settings.width, settings.height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDrawPixels()")) return GL_FALSE;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return GL_TRUE;
}

void QVSWidget::mousePressEvent(QMouseEvent *event)
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

void QVSWidget::mouseMoveEvent(QMouseEvent *event)
{
    float dx = MOUSE_SCALE * (event->x() - mouseX) / (float)settings.width;
    float dy = MOUSE_SCALE * (event->y() - mouseY) / (float)settings.height;

    QVector3 mouse(dx, -dy, 0.0);
    QVector3 view(0.0, 0.0, -1.0);
    QVector3 rotateAxis = QVector3::normalize(QVector3::cross(mouse, view));

    switch (mouseMode)
    {
    case MOUSE_DOLLY:
        settings.viewTranslation.z += dy;
        break;
    case MOUSE_ROTATE:
        settings.viewRotation = QVector4::normalize(QVector4::fromAngleAxis(dx * dx + dy * dy, rotateAxis) * settings.viewRotation);
        break;
    case MOUSE_TRANSLATE:
        settings.viewTranslation = settings.viewTranslation + mouse;
        break;
    default:
        break;
    }

    mouseX = event->x();
    mouseY = event->y();

    updateGL();
}

void QVSWidget::wheelEvent(QWheelEvent *event)
{
    updateGL();
}

// keyPressEvent
void QVSWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Escape:
    case Qt::Key_Q:
        // close();
        break;
    case Qt::Key_Plus:
        settings.volumeStepSize += STEPSIZE_DELTA;
        if (settings.volumeStepSize > STEPSIZE_MAX) settings.volumeStepSize = STEPSIZE_MAX;
        break;
    case Qt::Key_Minus:
        settings.volumeStepSize -= STEPSIZE_DELTA;
        if (settings.volumeStepSize < STEPSIZE_MIN) settings.volumeStepSize = STEPSIZE_MIN;
        break;
    case Qt::Key_Comma:
        settings.volumeStepSize *= 2.0;
        if (settings.volumeStepSize > STEPSIZE_MAX) settings.volumeStepSize = STEPSIZE_MAX;
        break;
    case Qt::Key_Period:
        settings.volumeStepSize *= 0.5;
        if (settings.volumeStepSize < STEPSIZE_MIN) settings.volumeStepSize = STEPSIZE_MIN;
        break;
    case Qt::Key_S:
        slotSaveConfigurations();
        break;
    case Qt::Key_L:
        slotLoadConfigurations();
        break;
    case Qt::Key_P:
        printSettings();
        break;
    case Qt::Key_Space:
        settings.enablePrintingFPS = !settings.enablePrintingFPS;
        break;
    }

    updateGL();
}

void QVSWidget::printSettings()
{
    std::cerr << " > LOG: step size " << settings.volumeStepSize << "." << std::endl;
    std::cerr << " > LOG: volume offset " << settings.volumeOffset << "." << std::endl;
    std::cerr << " > LOG: volume scale " << settings.volumeScale << "." << std::endl;
    std::cerr << " > LOG: print frames per second " << settings.enablePrintingFPS << "." << std::endl;
    std::cerr << " > LOG: light position (" << settings.lightDirection.x << ", " << settings.lightDirection.y << ", " << settings.lightDirection.z << std::endl;
    std::cerr << " > LOG: view rotation (" << settings.viewRotation.x << ", " << settings.viewRotation.y << ", " << settings.viewRotation.z << ", " << settings.viewRotation.w << ")." << std::endl;
    std::cerr << " > LOG: view translation (" << settings.viewTranslation.x << ", " << settings.viewTranslation.y << ", " << settings.viewTranslation.z << ")." << std::endl;
    std::cerr << " > LOG: window size (" << settings.width << ", " << settings.height << ")." << std::endl;
}

void QVSWidget::saveSettings(const std::string &name)
{
    std::ofstream file(name.c_str(), std::ios::binary);
    if (!file) return;

    if (!QSerializerT<QVector3>::write(file, settings.viewTranslation)) return;
    if (!QSerializerT<QVector4>::write(file, settings.viewRotation)) return;
    
    file.close();
}

void QVSWidget::loadSettings(const std::string &name)
{
    std::ifstream file(name.c_str(), std::ios::binary);
    if (!file) return;

    if (!QSerializerT<QVector3>::read(file, settings.viewTranslation)) return;
    if (!QSerializerT<QVector4>::read(file, settings.viewRotation)) return;

    file.read((char*)viewEntropy.data(), viewEntropy.size() * sizeof(float));
    file.close();
}

void QVSWidget::saveViewEntropy(const std::string &name)
{
    std::ofstream file(name.c_str(), std::ios::binary);
    if (!file) return;

    if (!QSerializerT<size_t>::write(file, viewSize)) return;
    file.write((char*)viewEntropy.data(), viewEntropy.size() * sizeof(float));
    file.close();
}

void QVSWidget::loadViewEntropy(const std::string &name)
{
    std::ifstream file(name.c_str(), std::ios::binary);
    if (!file) return;

    if (!QSerializerT<::size_t>::read(file, viewSize)) return;

    viewEntropy.resize(viewSize * viewSize * 2);
    file.read((char*)viewEntropy.data(), viewEntropy.size() * sizeof(float));
    file.close();

    ::size_t size = viewSize * viewSize * 2;
    float minEntropy(FLT_MAX), maxEntropy(0.0), *pEntropy(viewEntropy.data());
    for (int i = 0; i < size; i++)
    {
        float entropy = *(pEntropy++);
        if (entropy == 0.0f) continue;
        if (entropy > maxEntropy) maxEntropy = entropy;
        if (entropy < minEntropy) minEntropy = entropy;
    }

    emit signalNorthernViewEntropyInitialized(viewSize, viewEntropy.data(), minEntropy, maxEntropy);
    emit signalSouthernViewEntropyInitialized(viewSize, viewEntropy.data() + viewSize * viewSize, minEntropy, maxEntropy);
}

void QVSWidget::slotLoadConfigurations()
{
    loadSettings();
    loadViewEntropy();

    emit signalLoadViewEntropy();
}

void QVSWidget::slotSaveConfigurations()
{
    saveSettings();
    saveViewEntropy();

    emit signalSaveViewEntropy();
}

void QVSWidget::slotComputeViewEntropy()
{
    makeCurrent();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    QVector4 angleAxis;
    float scale = 2.0f / viewSize;
    int halfSize((viewSize + 1) / 2);
    for (int i = 0; i < viewSize; i++)
    {
        std::cerr << " > LOG: slotComputeViewEntropy() - " << i + 1 << std::endl;

        for (int j = 0; j < halfSize; j++)
        {
            for (int zsign = 0; zsign < 2; zsign++)
            {
                for (int ysign = 0; ysign < 2; ysign++)
                {
                    float nx((i + 0.5f) * scale - 1.0f), ny((j + 0.5f) * scale - 1.0f);
                    float nz(1.0f - nx * nx - ny * ny);
                    if (nz < 0.0f) continue;

                    float qx(std::sqrt((1 - (zsign ? 1 : -1) * std::sqrt(nz)) * 0.5f));
                    float qy(0.0f);
                    float qz(qx < EPSILON ? 0.0f : 0.5f * nx / qx);
                    float qw((ysign ? -1 : 1) * std::sqrt(1.0f - qx * qx - qy * qy - qz * qz));
                    QVector4 viewRotation(qx, qy, qz, qw);

                    glFlush();

                    glLoadIdentity();
                    QVector4::getAngleAxis(viewRotation, angleAxis);
                    glRotatef(-angleAxis.w * 180.0 / PI, angleAxis.x, angleAxis.y, angleAxis.z);
                    glTranslatef(-settings.viewTranslation.x, -settings.viewTranslation.y, -settings.viewTranslation.z);
                    glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *)inverseViewMatrix.data());

                    computeHistograms(GL_TRUE);
                    computeEntropy(viewRotation, GL_FALSE);
                }
            }
        }
    }

    glPopMatrix();

    slotSaveConfigurations();
    slotLoadConfigurations();
}

void QVSWidget::markPoint(int type)
{
    QVector4 rotation(settings.viewRotation);
    QVector3 normal(
        2.0f * (rotation.x * rotation.z + rotation.y * rotation.w),
        2.0f * (rotation.y * rotation.z - rotation.x * rotation.w),
        1.0f - 2.0f * (rotation.x * rotation.x + rotation.y * rotation.y)
        );

    float viewScale = (viewSize - 1) * 0.5;
    if (normal.z >= 0)
    {
        float x(viewScale * (-normal.x + 1.0f)), y(viewScale * (normal.y + 1.0f));
        emit signalNorthernViewPointMarked(type, (int)x + (int)y * viewSize);
        emit signalSouthernViewPointMarked(type, -1);
    }
    else
    {
        float x(viewScale * (normal.x + 1.0f)), y(viewScale * (-normal.y + 1.0f));
        emit signalNorthernViewPointMarked(type, -1);
        emit signalSouthernViewPointMarked(type, (int)x + (int)y * viewSize);
    }
}

void QVSWidget::slotMarkStartPoint()
{
    markPoint(0);
}

void QVSWidget::slotMarkEndPoint()
{
    markPoint(1);
}

std::string QVSWidget::getOptions()
{
    std::stringstream options;

#ifdef __CL_ENABLE_DEBUG
    options << "-D __CL_ENABLE_DEBUG ";
#endif
    if (settings.enableComputingEntropy) options << "-D __CL_ENABLE_COMPUTING_ENTROPY ";
    if (settings.enableShading) options << "-D __CL_ENABLE_SHADING ";
    options << "-D GAUSSIAN_1D=" << settings.gaussian1D << " ";
    options << "-D GAUSSIAN_2D=" << settings.gaussian2D << " ";

    return options.str();
}