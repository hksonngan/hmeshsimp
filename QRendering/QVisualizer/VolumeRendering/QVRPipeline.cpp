/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVRPipeline.cpp
 * @brief   QVRReader class, QVRPreprocessor class and QVRWriter class declaration.
 * 
 * This file declares the commonly used methods defined in QVRPipeline.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <iomanip>
#include <iostream>
#include <sstream>

#include <QMutex>
#include <QWaitCondition>
#include <QDateTime>

#include "../utilities/QIO.h"
#include "../utilities/QUtility.h"
#include "../infrastructures/QPipeline.h"
#include "../infrastructures/QCLMemory.h"
#include "QVRWidget.h"
#include "QVRSetting.h"
#include "QVRPipeline.h"

QVRReader::QVRReader(QVRWidget* parent) : QStage(parent),
    parent(parent)
{}

QVRReader::~QVRReader()
{}

void QVRReader::init(const std::string& name)
{
    this->name = name;
}

void QVRReader::run()
{
    cl_uint width = 0;
    for (::size_t quotient = parent->timeSteps; quotient > 0; quotient /= 10) width++;

    while (true)
    {
        cl_bool ready(CL_FALSE);
        cl_uint step(0), index(0), map(0);

        parent->statusMutex->lock();
        ::size_t stepEnd = min(parent->settings->currentStep + parent->cacheSize, parent->timeSteps);
        for (step = parent->settings->currentStep; step < stepEnd; step++)
        {
            index = step - parent->settings->currentStep;
            map = parent->cacheMapping.at(index);
            ready = parent->cacheStatus.at(index) == QCL_INITIALIZED;
            if (ready) break;
        }
        if (!ready) parent->readingFinished->wait(parent->statusMutex);
        parent->statusMutex->unlock();
        if (!ready) continue;

        std::stringstream info(std::stringstream::in | std::stringstream::out);
        info << "QVRReader::run() - " << step << " [" << map << "]";
        std::cerr << " > LOG: " << info.str() << " started." << std::endl;

        QDateTime start = QDateTime::currentDateTime();
        std::stringstream stepStream(std::stringstream::in | std::stringstream::out);
        stepStream << std::setw(width) << std::setfill('0') << (step + 1);
        float* ptrVolume = (float*)parent->cacheVolumeData.data() + map * parent->cacheVolumeSize;
        ::size_t voxelSize = 0;
        switch (parent->format)
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
        parent->volumeMutex[map].lock();
        QIO::getFileData(parent->timeSteps == 1 ? name : name + stepStream.str(), ptrVolume, parent->cacheVolumeSize * voxelSize);
        parent->volumeMutex[map].unlock();

        std::cerr << " > LOG: " << info.str() << " finished." << std::endl;
        ::size_t size = parent->cacheVolumeSize * voxelSize;
        if (parent->settings->enablePrintingBandwidth) QUtility::printBandWidth(size, start.msecsTo(QDateTime::currentDateTime()), info.str());

        parent->statusMutex->lock();
        cl_uint currentIndex = step - parent->settings->currentStep;
        if (currentIndex >= 0 && currentIndex < parent->cacheSize && parent->cacheStatus.at(currentIndex) == QCL_INITIALIZED)
        {
            std::cerr << " > LOG: " << info.str() << " hited." << std::endl;
            parent->cacheStatus.at(currentIndex) = QCL_READ;
            parent->preprocessingFinished->wakeAll();
        }
        parent->statusMutex->unlock();
    }
}

QVRPreprocessor::QVRPreprocessor(QVRWidget* parent) : QStage(parent),
    parent(parent)
{}

QVRPreprocessor::~QVRPreprocessor()
{

}

void QVRPreprocessor::run()
{
    cl_uint width = 0;
    for (::size_t quotient = parent->timeSteps; quotient > 0; quotient /= 10) width++;

    while (true)
    {
        cl_bool ready(CL_FALSE);
        cl_uint step(0), index(0), map(0);

        parent->statusMutex->lock();
        ::size_t stepEnd = min(parent->settings->currentStep + parent->cacheSize, parent->timeSteps);
        for (step = parent->settings->currentStep; step < stepEnd; step++)
        {
            index = step - parent->settings->currentStep;
            map = parent->cacheMapping.at(index);
            ready = parent->cacheStatus.at(index) == QCL_READ;
            if (ready) break;
        }
        if (!ready) parent->preprocessingFinished->wait(parent->statusMutex);
        parent->statusMutex->unlock();
        
        if (!ready) continue;

        std::stringstream info(std::stringstream::in | std::stringstream::out);
        info << "QVRPreprocessor::run() - " << step << " [" << map << "]";
        std::cerr << " > LOG: " << info.str() << " started." << std::endl;

        QDateTime start = QDateTime::currentDateTime();
        float* ptrVolume = (float*)parent->cacheVolumeData.data() + map * parent->cacheVolumeSize;
        float* ptrHistogram = (float*)parent->cacheHistogramData.data() + map * parent->cacheHistogramSize;
        parent->volumeMutex[map].lock();
        QUtility::preprocess(ptrVolume, parent->cacheVolumeSize, parent->format, parent->endian, parent->cacheHistogramSize, ptrHistogram, parent->valueMin, parent->valueMax);
        parent->volumeMutex[map].unlock();

        std::cerr << " > LOG: " << info.str() << " finished." << std::endl;
        ::size_t size = (parent->cacheVolumeSize + parent->cacheHistogramSize) * sizeof(float);
        if (parent->settings->enablePrintingBandwidth) QUtility::printBandWidth(size, start.msecsTo(QDateTime::currentDateTime()), info.str());

        parent->statusMutex->lock();
        cl_uint currentIndex = step - parent->settings->currentStep;
        if (currentIndex >= 0 && currentIndex < parent->cacheSize && parent->cacheStatus.at(currentIndex) == QCL_READ)
        {
            std::cerr << " > LOG: " << info.str() << " hited." << std::endl;
            parent->cacheStatus.at(currentIndex) = QCL_PREPROCESSED;
            parent->writingFinished->wakeAll();
        }
        parent->statusMutex->unlock();
    }
}

QVRWriter::QVRWriter(QVRWidget* parent) : QStage(parent),
    parent(parent)
{}

QVRWriter::~QVRWriter()
{}

void QVRWriter::run()
{
    cl_uint width = 0;
    for (::size_t quotient = parent->timeSteps; quotient > 0; quotient /= 10) width++;

    while (true)
    {
        cl_bool ready(CL_FALSE);
        cl_uint step(0), index(0), map(0), clMap(0);

        parent->statusMutex->lock();
        ::size_t stepEnd = min(parent->settings->currentStep + parent->clCacheSize, parent->timeSteps);
        for (step = parent->settings->currentStep; step < stepEnd; step++)
        {
            index = step - parent->settings->currentStep;
            map = parent->cacheMapping.at(index);
            clMap = parent->clCacheMapping.at(index);
            ready = parent->cacheStatus.at(index) == QCL_PREPROCESSED;
            if (ready) break;
        }
        if (!ready) parent->writingFinished->wait(parent->statusMutex);
        parent->statusMutex->unlock();
        if (!ready) continue;

        std::stringstream info(std::stringstream::in | std::stringstream::out);
        info << "QVRWriter::run() - " << step << " [" << map << "]";
        std::cerr << " > LOG: " << info.str() << " started." << std::endl;

        QDateTime start = QDateTime::currentDateTime();
        std::vector<::size_t> bufferOrigin(3, 0), hostOrigin(3, 0), size(3, 1), pitch(2, 0);
        bufferOrigin.at(0) = clMap * parent->cacheVolumeSize * sizeof(float);
        hostOrigin.at(0) = map * parent->cacheVolumeSize * sizeof(float);
        size.at(0) = parent->cacheVolumeSize * sizeof(float);
        parent->volumeMutex[map].lock();
        parent->clCacheVolumeData->read(parent->clQueue, bufferOrigin, hostOrigin, size, pitch);
        parent->volumeMutex[map].unlock();

        std::cerr << " > LOG: " << info.str() << " finished." << std::endl;
        if (parent->settings->enablePrintingBandwidth) QUtility::printBandWidth(parent->cacheVolumeSize * sizeof(float), start.msecsTo(QDateTime::currentDateTime()), info.str());

        parent->statusMutex->lock();
        cl_uint currentIndex = step - parent->settings->currentStep;
        if (currentIndex >= 0 && currentIndex < parent->cacheSize && parent->cacheStatus.at(currentIndex) == QCL_PREPROCESSED)
        {
            std::cerr << " > LOG: " << info.str() << " hited." << std::endl;
            parent->cacheStatus.at(currentIndex) = QCL_WRITTEN;
            if (currentIndex == 0) parent->paintingFinished->wakeAll();
        }
        parent->statusMutex->unlock();
    }
}