/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QMTWidget.cpp
 * @brief   QMTWidget class declaration.
 * 
 * This file declares ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
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
#include "../infrastructures/QVTKModel.h"
#include "../infrastructures/QGridModel.h"
#include "../infrastructures/QHoverPoints.h"
#include "../infrastructures/QCLProgram.h"
#include "../infrastructures/QPipeline.h"
#include "QMTControlPanel.h"
#include "QMTWidget.h"

QMTWidget::QMTWidget(QWidget *parent)
    : QGLWidget(parent),
    dataFileName(), dataFilePath(), objectFileName(), // Data file
    endian(ENDIAN_LITTLE), volumeSize(), thickness(), format(DATA_UNKNOWN), hpSize(0), hpHeight(0), intenityLevel(256),
        valueMin(FLT_MAX), valueMax(-FLT_MAX), // Volumetric Data
    cacheVolumeSize(0), cacheVolumeData(0), // Memory Cache
#ifdef __CL_ENABLE_DEBUG
    cacheDebugData(0),
#endif
    initialized(CL_FALSE), windowWidth(1.0f), windowLevel(0.5f), error(GL_FALSE), settings(), // Configuration
    mouseMode(MOUSE_ROTATE), mouseX(0), mouseY(0), bufferSize(0), totalNumber(0), isoValue(0.0f), isoSurfaceGenerated(GL_FALSE), // OpenGL Context
    clPrograms(), clDevices(), glVBO(0), clContext(0), clQueue(0) // OpenCL Context
{}

QMTWidget::~QMTWidget()
{
    this->destroy();
}

unsigned char QMTWidget::destroy()
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
unsigned char QMTWidget::initConnections(QMTControlPanel* panel)
{
    this->panel = panel;

    const Ui::QMTControlPanel* ui = panel->getUI();
    connect(ui->horizontalSliderIsoValue, SIGNAL(valueChanged(int)), this, SLOT(slotUpdateIsoValue()));

    settings.isoValue = (cl_float)ui->horizontalSliderIsoValue->value() / ui->horizontalSliderIsoValue->maximum();

    return GL_TRUE;
}

// Step 2 - init data
unsigned char QMTWidget::initData(const std::string &name)
{
    dataFileName = name;
    int position = dataFileName.find_last_of("\\");
    if (position == std::string::npos) position = dataFileName.find_last_of("/");
    if (position == std::string::npos) position = dataFileName.size() - 1;
    dataFilePath = dataFileName.substr(0, position + 1);

    if (error = !parseDataFile(dataFileName)) return GL_FALSE;
    
    return GL_TRUE;
}

unsigned char QMTWidget::parseDataFile(const std::string &name)
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
void QMTWidget::initContext()
{
    if (error |= !QUtility::checkSupport()) return;
    if (error |= !initOpenCL()) return;
    if (error |= !initConfigurations()) return;
    if (error |= !initPrograms()) return;
    if (error |= !initArguments()) return;

    this->initialized = true;
}

unsigned char QMTWidget::initOpenCL()
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

unsigned char QMTWidget::initConfigurations()
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

    hpHeight = 1;
    for (hpSize = 1; hpSize < maxSize; hpSize *= 2) hpHeight++;
    cacheVolumeSize = volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2];
    if (error = cacheVolumeData.size() < cacheVolumeSize * sizeof(float))
    {
        std::cerr << " > ERROR: limited volume memory allocated." << std::endl;
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

unsigned char QMTWidget::initPrograms()
{
    cl_int status = CL_SUCCESS;
    
    std::stringstream options;
    options << "-D HP_HEIGHT=" << hpHeight << " ";

    clPrograms.push_front(QCLProgram("Marching Cubes", "./cl/MarchingCubes/"));
    std::list<QCLProgram>::iterator pMarchingCubes = clPrograms.begin();
    pMarchingCubes->kernels.push_back(QCLKernel(pMarchingCubes->path, "Cube Classifier", "classifyCubes", "cube_classifier.cl"));
    pMarchingCubes->kernels.push_back(QCLKernel(pMarchingCubes->path, "HP Constructor", "constructHP", "hp_constructor.cl"));
    pMarchingCubes->kernels.push_back(QCLKernel(pMarchingCubes->path, "HP Traverser", "traverseHP", "hp_traverser.cl"));
    if (!pMarchingCubes->initialize(clContext, clDevices, options.str())) return GL_FALSE;

    // Create images for the HistogramPyramid
    cl_uint4 size = volumeSize;
    size.s[3] = sizeof(cl_float);
    pMarchingCubes->memories.push_front(QCLMemory("Volume Data", QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_ONLY, size));
    std::list<QCLMemory>::iterator mVolumeData = pMarchingCubes->memories.begin();
    if (!mVolumeData->initialize(clContext, std::vector<::size_t>(1, mVolumeData->bufferFormat.s[0] * mVolumeData->bufferFormat.s[1] * mVolumeData->bufferFormat.s[2] * mVolumeData->bufferFormat.s[3]), cacheVolumeData.data())) return GL_FALSE;

    size.s[3] = sizeof(cl_uchar);
    std::vector<cl_uint4> bufferFormat(hpHeight, size);
    for (int i = 1; i < hpHeight; i++)
    {
        for (int j = 0; j < 3; j++)
            bufferFormat.at(i).s[j] = (bufferFormat.at(i - 1).s[j] - 1) / 2 + 1;
        if (i < 2)
            bufferFormat.at(i).s[3] = sizeof(cl_uchar);
        else if (i < 5)
            bufferFormat.at(i).s[3] = sizeof(cl_ushort);
        else
            bufferFormat.at(i).s[3] = sizeof(cl_int);
    }
    
    for (int i = 0; i < hpHeight; i++)
    {
        std::stringstream name;
        ::size_t height = hpHeight - 1 - i;
        name << "HP Level-" << height;
        pMarchingCubes->memories.push_front(QCLMemory(name.str(), QCLMemory::QCL_BUFFER, CL_FALSE, CL_FALSE, CL_MEM_READ_WRITE, bufferFormat.at(height)));
        std::list<QCLMemory>::iterator mHPData = pMarchingCubes->memories.begin();
        if (!mHPData->initialize(clContext, std::vector<::size_t>(1, mHPData->bufferFormat.s[0] * mHPData->bufferFormat.s[1] * mHPData->bufferFormat.s[2] * mHPData->bufferFormat.s[3]))) return GL_FALSE;
    }

    return GL_TRUE;
}

unsigned char QMTWidget::initArguments()
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
    /*
    QVTKModel vtkModel;
    vtkModel.loadBinaryFile("E:/88Datasets/Tokyo/Tanaka/mergef_wg01--0100.bin");
    vtkModel.saveRawFiles("E:/88Datasets/Tokyo/Tanaka/mergef_wg01--0100");
    */

    cl_uint vexlNumber = volumeSize.s[0] * volumeSize.s[1] * volumeSize.s[2];
    QIO::getFileData(dataFilePath + objectFileName, cacheVolumeData.data(), voxelSize * vexlNumber);

    float volumeMin = FLT_MAX, volumeMax = FLT_MIN;
    QUtility::preprocess(cacheVolumeData.data(), vexlNumber, format, endian, 0, NULL, volumeMin, volumeMax);
    
    std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "Marching Cubes");
    std::list<QCLMemory>::iterator mVolumeData = QCLMemory::find(pMarchingCubes->memories, "Volume Data");
    mVolumeData->read(clQueue);

    return GL_TRUE;
}

// Step 4 - message loop

// public slots

unsigned char QMTWidget::slotUpdateIsoValue()
{
    if (!this->initialized) return GL_FALSE;

    cl_float isoValue = (cl_float)panel->getUI()->horizontalSliderIsoValue->value() / panel->getUI()->horizontalSliderIsoValue->maximum();
    if (isoValue != settings.isoValue)
    {
        settings.isoValue = isoValue;
        updateGL();
    }
    return GL_TRUE;
}

// initializeGL
void QMTWidget::initializeGL()
{
    initContext();

	glEnable(GL_NORMALIZE);
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	// Set material properties which will be assigned by glColor
	GLfloat color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
	GLfloat specReflection[] = { 0.8f, 0.8f, 0.8f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specReflection);
	GLfloat shininess[] = { 16.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

	// Create light components
	GLfloat ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	GLfloat diffuseLight[] = { 0.7f, 0.7f, 0.7f, 1.0f };
	GLfloat specularLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat position[] = { -0.0f, 4.0f, 1.0f, 1.0f };

	// Assign created components to GL_LIGHT0
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT0, GL_POSITION, position);
}

// resizeGL
void QMTWidget::resizeGL(int w, int h)
{
    if (error) return;

    settings.width = w == 0 ? 1 : w;
    settings.height = h == 0 ? 1 : h;

	glViewport(0, 0,  settings.width, settings.height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)settings.width / settings.height, 0.5f, 10000.0f);

	updateGL();
}

// paintGL
void QMTWidget::paintGL()
{
    if (error) return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(settings.viewTranslation.x, settings.viewTranslation.y, settings.viewTranslation.z);
	QVector4 rotation;
	QVector4::getAngleAxis(settings.viewRotation, rotation);
	glRotatef(rotation.w * 180.0 / PI, rotation.x, rotation.y, rotation.z);
	
    QDateTime tPaintGL = QDateTime::currentDateTime();
    
	if (isoValue != settings.isoValue)
	{
		isoValue = settings.isoValue;
        isoSurfaceGenerated = GL_FALSE;

		cl_int status = CL_SUCCESS;

        // Compute the number of triangles for each cube
        QDateTime tClassifyCubes = QDateTime::currentDateTime();
        if (!classifyCubes()) return;
        if (settings.enablePrintingFPS) QUtility::printTimeCost(tClassifyCubes.msecsTo(QDateTime::currentDateTime()), "classifyCubes()");

		// Construct the histogram pyramid according to iso-value
        QDateTime tConstructHP = QDateTime::currentDateTime();
		if (!constructHP()) return;
		if (settings.enablePrintingFPS) QUtility::printTimeCost(tConstructHP.msecsTo(QDateTime::currentDateTime()), "constructHP()");

		// Compute the number of triangles according to the top of histogram pyramid
		std::stringstream name;
		name << "HP Level-" << hpHeight - 1;
		std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "Marching Cubes");
		std::list<QCLMemory>::iterator mVolumeData = QCLMemory::find(pMarchingCubes->memories, name.str());

        std::vector<unsigned char> buffer(mVolumeData->size.at(0), 0);
		status = clEnqueueReadBuffer(clQueue, mVolumeData->get(), CL_FALSE, 0,  buffer.size(), buffer.data(), 0, NULL, NULL);
		if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBuffer()")) return;

		status = clFinish(clQueue);
		if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFinish()")) return;

        if (mVolumeData->bufferFormat.s[3] == 1)
            totalNumber = *((unsigned char*)buffer.data());
        else if (mVolumeData->bufferFormat.s[3] == 2)
            totalNumber = *((unsigned short*)buffer.data());
        else
            totalNumber = *((unsigned int*)buffer.data());
		if (totalNumber == 0)
		{
			std::cout << " > INFO: there are no triangles for the iso-surface I = " << isoValue << "." << std::endl;
			return;
		}

        unsigned int triangleSize = 18 * sizeof(cl_float);
        unsigned int size = totalNumber * triangleSize;
        if (size > CACHE_CL_BUFFER_SIZE) size = CACHE_CL_BUFFER_SIZE / triangleSize * triangleSize;

        // Delete the old VBO
        if (bufferSize > 0 && bufferSize < size)
        {
            status = clReleaseMemObject(clVBO);
            if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseMemObject()")) return;
            glDeleteBuffers(1, &glVBO);
            if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDeleteBuffers()")) return;
            bufferSize = 0;
        }

        // Allocate a new VBO to represent the iso-surface
        if (bufferSize < size)
        {
            bufferSize = 1;
            while (bufferSize < size) bufferSize += bufferSize / 8 + 1;
            bufferSize = ((bufferSize - 1) / triangleSize + 1) * triangleSize;

            glGenBuffers(1, &glVBO);
            if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glGenBuffers()")) return;

            glBindBuffer(GL_ARRAY_BUFFER, glVBO);
            glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_STATIC_DRAW);
            glVertexPointer(3, GL_FLOAT, 24, (char *)NULL + 0);
            glNormalPointer(GL_FLOAT, 24, (char *)NULL + 12);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            clVBO = clCreateFromGLBuffer(clContext, CL_MEM_WRITE_ONLY, glVBO, &status);
            if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateFromGLBuffer()")) return;
        }
	}

	GLfloat scalingFactor = 1.0f / hpSize;
	glColor3f(1.0f, 1.0f, 1.0f);
	glScalef(scalingFactor, scalingFactor, scalingFactor);
	glTranslatef(-0.5f * volumeSize.s[0], -0.5f * volumeSize.s[1], -0.5 * volumeSize.s[2]);

	if (bufferSize > 0)
	{
        // Normal Buffer
        glBindBuffer(GL_ARRAY_BUFFER, glVBO);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        unsigned int triangleSize = 18 * sizeof(cl_float);
        unsigned int size = bufferSize / triangleSize;
        if (totalNumber <= size && isoSurfaceGenerated)
        {
            glDrawArrays(GL_TRIANGLES, 0, size * 3);
            glFinish();
        }
        else
        {
            isoSurfaceGenerated = GL_TRUE;
            QDateTime tTraverseHP = QDateTime::currentDateTime();
            if (!traverseHP(size)) return;
            if (settings.enablePrintingFPS) QUtility::printTimeCost(tTraverseHP.msecsTo(QDateTime::currentDateTime()), "traverseHP()");
        }

        // Release buffer
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
    }

    if (settings.enablePrintingFPS) QUtility::printFPS(tPaintGL.msecsTo(QDateTime::currentDateTime()), "paintGL()");

    // QMetaObject::invokeMethod(this, "updateGL", Qt::QueuedConnection);
}

unsigned char QMTWidget::classifyCubes()
{
	cl_int status = CL_SUCCESS;

    std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "Marching Cubes");
    std::list<QCLKernel>::iterator kCubeClassifier = QCLKernel::find(pMarchingCubes->kernels, "Cube Classifier");
    std::list<QCLMemory>::iterator mHPData = QCLMemory::find(pMarchingCubes->memories, "HP Level-0");
    status |= clSetKernelArg(kCubeClassifier->get(), 0, sizeof(cl_mem), &mHPData->get());
    std::list<QCLMemory>::iterator mVolumeData = QCLMemory::find(pMarchingCubes->memories, "Volume Data");
    status |= clSetKernelArg(kCubeClassifier->get(), 1, sizeof(cl_mem), &mVolumeData->get());
    status |= clSetKernelArg(kCubeClassifier->get(), 2, sizeof(cl_float), &isoValue);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

	std::vector<::size_t> gridSize(volumeSize.s, volumeSize.s + 3);
	status = clEnqueueNDRangeKernel(clQueue, kCubeClassifier->get(), gridSize.size(), NULL, gridSize.data(), NULL, 0, NULL, NULL);
	if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

#ifdef __CL_ENABLE_DEBUG
    ::size_t bufferSize = min(cacheDebugData.size(), mHPData->size.at(0));
    status = clEnqueueReadBuffer(clQueue, mHPData->get(), CL_TRUE, 0, bufferSize, cacheDebugData.data(), 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBuffer()")) return GL_FALSE;
    cl_uchar* ptr = (cl_uchar*)cacheDebugData.data();
    std::cerr << " > LOG: " << bufferSize << std::endl;
#endif
	return GL_TRUE;
}

unsigned char QMTWidget::constructHP()
{
    cl_int status = CL_SUCCESS;

    std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "Marching Cubes");
    std::list<QCLKernel>::iterator kHPConstructor = QCLKernel::find(pMarchingCubes->kernels, "HP Constructor");
    std::list<QCLMemory>::iterator mHPData = QCLMemory::find(pMarchingCubes->memories, "HP Level-0");
    for(int i = 0; i < hpHeight - 1; i++)
    {
        status |= clSetKernelArg(kHPConstructor->get(), 0, sizeof(cl_uint4), &mHPData->bufferFormat);
        status |= clSetKernelArg(kHPConstructor->get(), 1, sizeof(cl_mem),  &mHPData->get());
        mHPData++;
        status |= clSetKernelArg(kHPConstructor->get(), 2, sizeof(cl_uint4), &mHPData->bufferFormat);
        status |= clSetKernelArg(kHPConstructor->get(), 3, sizeof(cl_mem),  &mHPData->get());
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

        std::vector<::size_t> gridSize(mHPData->bufferFormat.s, mHPData->bufferFormat.s + 3);
        status = clEnqueueNDRangeKernel(clQueue, kHPConstructor->get(), 3, NULL, gridSize.data(), NULL, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

#ifdef __CL_ENABLE_DEBUG
        ::size_t bufferSize = min(cacheDebugData.size(), mHPData->size.at(0));
        status = clEnqueueReadBuffer(clQueue, mHPData->get(), CL_TRUE, 0, bufferSize, cacheDebugData.data(), 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBuffer()")) return GL_FALSE;
        if (mHPData->bufferFormat.s[3] == 1)
        {
            cl_uchar* ptr = (cl_uchar*)cacheDebugData.data();
            std::cerr << " > LOG: " << bufferSize << std::endl;
        }
        else if (mHPData->bufferFormat.s[3] == 2)
        {
            cl_ushort* ptr = (cl_ushort*)cacheDebugData.data();
            std::cerr << " > LOG: " << bufferSize << std::endl;
        }
        else if (mHPData->bufferFormat.s[3] == 4)
        {
            cl_uint* ptr = (cl_uint*)cacheDebugData.data();
            std::cerr << " > LOG: " << bufferSize << std::endl;
        }
#endif
    }

    return GL_TRUE;
}

unsigned char QMTWidget::traverseHP(cl_uint size)
{
	cl_int status = CL_SUCCESS;

	std::list<QCLProgram>::iterator pMarchingCubes = QCLProgram::find(clPrograms, "Marching Cubes");
	std::list<QCLKernel>::iterator kHPTraverser = QCLKernel::find(pMarchingCubes->kernels, "HP Traverser");

    int index = 0;
    std::list<QCLMemory>::iterator mHPData = QCLMemory::find(pMarchingCubes->memories, "HP Level-0");
    while(index < hpHeight * 2)
    {
        status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_mem), &mHPData->get());
        status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_uint4), &mHPData->bufferFormat);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;
        mHPData++;
    }

    status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_mem), &clVBO);
    std::list<QCLMemory>::iterator mVolumeData = QCLMemory::find(pMarchingCubes->memories, "Volume Data");
    status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_mem), &mVolumeData->get());
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_float), &isoValue);
    status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_uint), &totalNumber);
    status |= clSetKernelArg(kHPTraverser->get(), index++, sizeof(cl_uint), &hpHeight);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

    status = clEnqueueAcquireGLObjects(clQueue, 1, &clVBO, 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueAcquireGLObjects()")) return GL_FALSE;
    
    std::vector<::size_t> gridSize(1, size);
    unsigned int totalSize = 0;
    while (totalSize < totalNumber)
    {
        status |= clSetKernelArg(kHPTraverser->get(), index, sizeof(cl_uint), &totalSize);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clSetKernelArg()")) return GL_FALSE;

        // Run a NDRange kernel over this buffer which traverses back to the base level
        status = clEnqueueNDRangeKernel(clQueue, kHPTraverser->get(), gridSize.size(), NULL, gridSize.data(), NULL, 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return GL_FALSE;

#ifdef __CL_ENABLE_DEBUG
        ::size_t bufferSize = min(cacheDebugData.size(), size * 18 * sizeof(float));
        status = clEnqueueReadBuffer(clQueue, clVBO, CL_TRUE, 0, bufferSize, cacheDebugData.data(), 0, NULL, NULL);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReadBuffer()")) return GL_FALSE;
        float* ptr = (float*)cacheDebugData.data();
        std::cerr << " > LOG: " << bufferSize << std::endl;
#endif
        
        status = clFinish(clQueue);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clFinish()")) return GL_FALSE;
        
        //QDateTime tDrawArrays = QDateTime::currentDateTime();
        glDrawArrays(GL_TRIANGLES, 0, size * 3);
        glFinish();
        //if (settings.enablePrintingFPS) QUtility::printTimeCost(tDrawArrays.msecsTo(QDateTime::currentDateTime()), "glDrawArrays()");
        
        totalSize += size;
    }

    status = clEnqueueReleaseGLObjects(clQueue, 1, &clVBO, 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueReleaseGLObjects()")) return GL_FALSE;
    
	return GL_TRUE;
}

void QMTWidget::mousePressEvent(QMouseEvent *event)
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

void QMTWidget::mouseMoveEvent(QMouseEvent *event)
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

void QMTWidget::wheelEvent(QWheelEvent *event)
{
    updateGL();
}

// keyPressEvent
void QMTWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Escape:
    case Qt::Key_Q:
        // close();
        break;
    case Qt::Key_Plus:
        settings.isoValue += ISOVALUE_DELTA;
        if (settings.isoValue > ISOVALUE_MAX) settings.isoValue = ISOVALUE_MAX;
        break;
    case Qt::Key_Minus:
        settings.isoValue -= ISOVALUE_DELTA;
        if (settings.isoValue < ISOVALUE_MIN) settings.isoValue = ISOVALUE_MIN;
        break;
    case Qt::Key_Comma:
        settings.isoValue *= 2.0f;
        if (settings.isoValue > ISOVALUE_MAX) settings.isoValue = ISOVALUE_MAX;
        break;
    case Qt::Key_Period:
        settings.isoValue *= 0.5f;
        if (settings.isoValue < ISOVALUE_DELTA * 0.5f) settings.isoValue = ISOVALUE_MIN;
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
        settings.enablePrintingFPS = !settings.enablePrintingFPS;
        break;
    }

    updateGL();
}

void QMTWidget::saveSettings()
{}

void QMTWidget::loadSettings()
{}

void QMTWidget::printSettings()
{
    std::cerr << " > LOG: iso-value " << settings.isoValue << "." << std::endl;
    std::cerr << " > LOG: print frames per second " << settings.enablePrintingFPS << "." << std::endl;
    std::cerr << " > LOG: light direction (" << settings.lightDirection.x << ", " << settings.lightDirection.y << ", " << settings.lightDirection.z << std::endl;
    std::cerr << " > LOG: view rotation (" << settings.viewRotation.x << ", " << settings.viewRotation.y << ", " << settings.viewRotation.z << ", " << settings.viewRotation.w << ")." << std::endl;
    std::cerr << " > LOG: view translation (" << settings.viewTranslation.x << ", " << settings.viewTranslation.y << ", " << settings.viewTranslation.z << ")." << std::endl;
    std::cerr << " > LOG: window size (" << settings.width << ", " << settings.height << ")." << std::endl;
}
