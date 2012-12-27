/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    *.cl
 * @brief   * functions definition.
 * 
 * This file defines *.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

int computeFearures(
    const float counterScaling,
    const float valueScaling,
    const int quantizationLevel,
    __global float *lFearure,
    __local int *lIndices,
    __local int *lCounters)
{
    int lNumber = 0;
    for (int i = 0; i < quantizationLevel; i++)
    {
        if (lCounters[i] > 0) lIndices[lNumber++] = i;
    }

    // compute the mean value of intensity histogram.
    float mean = 0.0f;
    float scaling = counterScaling * valueScaling;
    for (int i = 0; i < lNumber; i++)
    {
        int value = lIndices[i];
        mean += (float)lCounters[value] * value * scaling;
    }

    // compute the standard deviation, relative smoothness, third moment, uniformity, entropy and gradient.
    float tStandardDeviation = 0.0f;
    //float tUniformity = 0.0f;
    //float tEntropy = 0.0f;
    //float tThirdMoment = 0.0f;
    for (int i = 0; i < lNumber; i++)
    {
        int value = lIndices[i];
        float normalizedValue = value * valueScaling;
        float density = (float)lCounters[value] * counterScaling;
        tStandardDeviation += (normalizedValue - mean) * (normalizedValue - mean) * density;
        //tThirdMoment += (normalizedValue - mean) * (normalizedValue - mean) * (normalizedValue - mean) * density;
        //tUniformity += density * density;
        //tEntropy += -density * (float)(log(density) / log(2.0f));
    }

    // feature 1: intensity
    lFearure[0] = (float)sqrt(tStandardDeviation);

    //// feature 2: relative smoothness
    //pVolumeSmoothness[gi] = 1.0f - 1.0f / (1.0f + volumeFeature[gi]);

    //// feature 3: uniformaty
    //pVolumeUniformityormaty[gi] = (float)tUniformityormity;

    //// feature 4: entropy
    //pVolumeEntropy[gi] = (float)tEntropy;

    //// feature 5: third-moment
    //pVolumeThirdMoment[gi] = (float)tThirdMoment;

    return lNumber;
}

__kernel void entrance_feature_extractor(
    const int quantizationLevel,
    const int neighborhoodSize,
    const int4 volumeSize,
    const int4 passSize,
    __global __read_only int *volumeData,
    __global __write_only float *volumeFeature,
    __local int *gIndices,
    __local int *gCounters
   )
{
    const float valueScaling = 1.0f / (quantizationLevel - 1);
    const int4 offset = (int4)(1, volumeSize.x, volumeSize.x * volumeSize.y, volumeSize.x * volumeSize.y * volumeSize.z);
    const int4 nStart = (int4)(neighborhoodSize) * offset;
    const int4 nEnd = nStart + offset;
    const int4 lID = (int4)(get_local_id(0), get_local_id(1), get_local_id(2), 0);
    const int4 lSize = (int4)(get_local_size(0), get_local_size(1), get_local_size(2), 1);
    int lOffset = lID.x + lID.y * lSize.x + lID.z * lSize.x * lSize.y;
    __local int *lIndices = gIndices + quantizationLevel * lOffset;
    __local int *lCounters = gCounters + quantizationLevel * lOffset;
    for (int i = 0; i < quantizationLevel; i++)
    {
        lIndices[i] = 0;
        lCounters[i] = 0;
    }
    
    int lNumber = 0;
    int volNumber = 0;
    int4 gIndex = (int4)0;
    const long4 gID = (long4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    const long4 gSize = (long4)(get_global_size(0), get_global_size(1), get_global_size(2), 1);
    const long4 gOffset = as_long4(convert_long4(offset));
    const int4 gStart = convert_int4(convert_long4(passSize) * gID / gSize * gOffset);
    const int4 gEnd = convert_int4(convert_long4(passSize) * (gID + (long4)1) / gSize * gOffset);
    for (gIndex.z = gStart.z; gIndex.z < gEnd.z; gIndex.z += offset.z)
    {
        for (gIndex.y = gStart.y; gIndex.y < gEnd.y; gIndex.y += offset.y)
        {
            gIndex.x = gStart.x;
            for (int i = 0; i < lNumber; i++)
            {
                lCounters[lIndices[i]] = 0;
                lIndices[i] = 0;
            }
            volNumber = 0;

            int gi = gIndex.z + gIndex.y + gIndex.x;
            int4 lStart, lEnd, lIndex;
            lStart.z = gi - min(nStart.z, gIndex.z);
            lEnd.z = gi + min(nEnd.z, offset.w - gIndex.z);
            for (lIndex.z = lStart.z; lIndex.z < lEnd.z; lIndex.z += offset.z)
            {
                lStart.y = lIndex.z - min(nStart.y, gIndex.y);
                lEnd.y = lIndex.z + min(nEnd.y, offset.z - gIndex.y);
                for (lIndex.y = lStart.y; lIndex.y < lEnd.y; lIndex.y += offset.y)
                {
                    lStart.x = lIndex.y - min(nStart.x, gIndex.x);
                    lEnd.x = lIndex.y + min(nEnd.x, offset.y - gIndex.x);
                    volNumber += lEnd.x - lStart.x;
                    for (lIndex.x = lStart.x; lIndex.x < lEnd.x; lIndex.x += offset.x)
                    {
                        lCounters[volumeData[lIndex.x]]++;
                    }
                }
            }

            __global float *lFearure = volumeFeature + gi;
            lNumber = computeFearures(1.0f / volNumber, valueScaling, quantizationLevel, lFearure, lIndices, lCounters);

            for (gIndex.x += offset.x; gIndex.x < gEnd.x; gIndex.x += offset.x)
            {
                int4 lStart, lEnd, lIndex;
                if (offset.y - gIndex.x >= nEnd.x)
                {
                    lStart.z = gi + nEnd.x - min(nStart.z, gIndex.z);
                    lEnd.z = gi + nEnd.x + min(nEnd.z, offset.w - gIndex.z);
                    for (lIndex.z = lStart.z; lIndex.z < lEnd.z; lIndex.z += offset.z)
                    {
                        lStart.y = lIndex.z - min(nStart.y, gIndex.y);
                        lEnd.y = lIndex.z + min(nEnd.y, offset.z - gIndex.y);
                        for (lIndex.y = lStart.y; lIndex.y < lEnd.y; lIndex.y += offset.y)
                        {
                            volNumber++;
                            lCounters[volumeData[lIndex.y]]++;
                        }
                    }
                }
                
                if (gIndex.x - offset.x >= nStart.x)
                {
                    lStart.z = gi - nStart.x - min(nStart.z, gIndex.z);
                    lEnd.z = gi - nStart.x + min(nEnd.z, offset.w - gIndex.z);
                    for (lIndex.z = lStart.z; lIndex.z < lEnd.z; lIndex.z += offset.z)
                    {
                        lStart.y = lIndex.z - min(nStart.y, gIndex.y);
                        lEnd.y = lIndex.z + min(nEnd.y, offset.z - gIndex.y);
                        for (lIndex.y = lStart.y; lIndex.y < lEnd.y; lIndex.y += offset.y)
                        {
                            volNumber--;
                            lCounters[volumeData[lIndex.y]]--;
                        }
                    }
                }
                
                gi += offset.x;
                lFearure += offset.x;
                lNumber = computeFearures(1.0f / volNumber, valueScaling, quantizationLevel, lFearure, lIndices, lCounters);
            }
        }
    }
}