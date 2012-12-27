/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    *.cpp
 * @brief   * class declaration.
 * 
 * This file declares *.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <iostream>

#include <QTime>

#include "../utilities/QIO.h"
#include "../utilities/QUtility.h"
#include "../infrastructures/QStructure.h"
#include "QVRFeatureExtraction.h"

unsigned char QVRFeatureExtracation::extractFeatures(const cl_context& clContext, std::list<QCLProgram>& clPrograms, cl_command_queue& clQueue, unsigned char* volumeData, unsigned int* volumeSize, int voxelSize)
{
    cl_int status = CL_SUCCESS;

    std::list<QCLProgram>::iterator pFeatureExtraction = QCLProgram::find(clPrograms, "Feature Extraction");
    std::list<QCLKernel>::iterator kFeatureExtractor = QCLKernel::find(pFeatureExtraction->kernels, "Feature Extractor");

    cl_int quantizationLevel = 256, neighborhoodSize = 3;
    status = clSetKernelArg(kFeatureExtractor->get(), 0, sizeof(cl_int), (void *)&quantizationLevel);
    status = clSetKernelArg(kFeatureExtractor->get(), 1, sizeof(cl_int), (void *)&neighborhoodSize);
    
    cl_int4 size = { volumeSize[0], volumeSize[1], volumeSize[2], 1 };
    status = clSetKernelArg(kFeatureExtractor->get(), 2, sizeof(cl_int4), &size);
    cl_int4 passSize = { size.s[0] >> 2, size.s[1] >> 2, size.s[2] >> 2, 1 };
    status = clSetKernelArg(kFeatureExtractor->get(), 3, sizeof(cl_int4), &passSize);
    
    int length = volumeSize[0] * volumeSize[1] * volumeSize[2];
    std::vector<int> data = std::vector<int>(length);
    std::vector<float> feature = std::vector<float>(length);
    int *ptrData = data.data();
    switch (voxelSize)
    {
    case 1:
        {
            float scale = (quantizationLevel - 1) / 255.0f;
            unsigned char *ptrVolume = (unsigned char *)volumeData;
            for (int i = 0; i < length; i++)
                *(ptrData++) = (int)(*(ptrVolume++) * scale);
        }
        break;
    case 2:
        {
            float scale = (quantizationLevel - 1) / 65535.0f;
            unsigned short *ptrVolume = (unsigned short *)volumeData;
            for (int i = 0; i < length; i++)
                *(ptrData++) = (int)(*(ptrVolume++) * scale);
        }
        break;
    case 4:
        {
            float scale = quantizationLevel - 1;
            float *ptrVolume = (float *)volumeData;
            for (int i = 0; i < length; i++)
                *(ptrData++) = (int)(*(ptrVolume++) * scale);
        }
        break;
    }

    cl_mem clData = clCreateBuffer(clContext, CL_MEM_READ_ONLY, data.size() * sizeof(int), NULL, &status);
    status = clSetKernelArg(kFeatureExtractor->get(), 4, sizeof(cl_mem), &clData);
    cl_mem clFeature = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, feature.size() * sizeof(float), NULL, &status);
    status = clSetKernelArg(kFeatureExtractor->get(), 5, sizeof(cl_mem), &clFeature);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateBuffer()")) return CL_FALSE;

    const int workGroupSize = 1 << 3, workItems = 1 << 1;
    status = clSetKernelArg(kFeatureExtractor->get(), 6, workItems * quantizationLevel * sizeof(cl_int), NULL);
    status = clSetKernelArg(kFeatureExtractor->get(), 7, workItems * quantizationLevel * sizeof(cl_int), NULL);

    std::vector<float> testing = std::vector<float>(length);
    QIO::getFileData("head_standard_deviation_for_test.raw", &testing[0], testing.size() * sizeof(float));

    QTime time = QTime::currentTime();

    status = clEnqueueWriteBuffer(clQueue, clData, CL_TRUE, 0, data.size() * sizeof(int), &data[0], 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueWriteBuffer()")) return CL_FALSE;

    std::cout << " > INFO: running CL program." << std::endl;
    const int localWorkSize = workItems,  globalWorkSize = workGroupSize * localWorkSize;
    
    const cl_uint dimensions = 3;
    const ::size_t global_work_size[] = { globalWorkSize, globalWorkSize, globalWorkSize };
    const ::size_t  local_work_size[] = {  localWorkSize,              1,              1 };
    status = clEnqueueNDRangeKernel(clQueue, kFeatureExtractor->get(), dimensions, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clEnqueueNDRangeKernel()")) return CL_FALSE;
    
    status = clEnqueueReadBuffer(clQueue, clFeature, CL_TRUE, 0, feature.size() * sizeof(float), &feature[0], 0, NULL, NULL);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "CommandQueue::enqueueReadBuffer()")) return CL_FALSE;

    status = clFinish(clQueue);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "Event::wait()")) return CL_FALSE;
    
    std::cerr << " > INFO: time elapsed - " << time.elapsed() << " ms." << std::endl;
    
    float *ptrFeature = &feature[0];
    float *ptrTesting = &testing[0];
    float maxError = 0.0f, minError = 1.0, totalError = 0.0f;
    int maxErrorID = 0.0f, minErrorID = 1.0;
    int sizeTesting = passSize.s[0] * passSize.s[1] * passSize.s[2];
    
    for (int i = 0; i < passSize.s[2]; i++)
    {
        for (int j = 0; j < passSize.s[1]; j++)
        {
            for (int k = 0; k < passSize.s[0]; k++)
            {
                int id = k + j * size.s[0] + i * size.s[0] * size.s[1];
                float error = std::abs(feature[id] - testing[id]);
                totalError += error;
                if (error > maxError)
                {
                    maxError = error;
                    maxErrorID = id;
                }
                if (error < minError)
                {
                    minError = error;
                    minErrorID = id;
                }
            }
        }
    }

    std::cerr << " > INFO: time elapsed - " << time.elapsed() << " ms." << std::endl;

    return CL_TRUE;
}