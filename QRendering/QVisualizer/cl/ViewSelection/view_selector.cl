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

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define EPSILON         1e-5

#if GAUSSIAN_1D == 7
    __constant float gaussian1[7] =
    {
        0.0548f, 0.1239f, 0.2022f, 0.2381f, 0.2022f, 0.1239f, 0.0548f
    };
#elif GAUSSIAN_1D == 5
    __constant float gaussian1[5] =
    {
        0.0924f, 0.2414f, 0.3324f, 0.2414f, 0.0924f
    };
#elif GAUSSIAN_1D == 3
    __constant float gaussian1[3] =
    {
        0.2256, 0.5488, 0.2256f
    };
#else
    __constant float gaussian1[1] =
    {
        1.0000f
    };
#endif

float getInterpolatedValue(__global float4 *data, const uint4 size, const uint4 offset, float4 position)
{
    const float4 indexf = floor(position);
    const float4 nextf = min(indexf + 1.0, convert_float4(size - 1));
    const uint4 index = convert_uint4(indexf);
    const uint4 next = convert_uint4(nextf);
    const float8 x = (float8)(
        data[index.x + index.y * offset.y + index.z * offset.z].w,
        data[index.x + index.y * offset.y +  next.z * offset.z].w,
        data[index.x +  next.y * offset.y + index.z * offset.z].w,
        data[index.x +  next.y * offset.y +  next.z * offset.z].w,
        data[ next.x + index.y * offset.y + index.z * offset.z].w,
        data[ next.x + index.y * offset.y +  next.z * offset.z].w,
        data[ next.x +  next.y * offset.y + index.z * offset.z].w,
        data[ next.x +  next.y * offset.y +  next.z * offset.z].w
    );

    const float4 a = position - convert_float4(index);
    const float4 y = mix(x.lo, x.hi, (float4)(a.x));
    const float2 z = mix(y.lo, y.hi, (float2)(a.y));
    const float  r = mix(z.lo, z.hi, (float) (a.z));
    return r;
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

void intersectRay(float16 r_o, float16 r_d, float16 box, float4 *tnear, float4 *tfar)
{
    // compute intersection of ray with all six bbox planes
    const float16 invR = (float16)(1.0f) / r_d;
    const float16 tbot = invR * (-box - r_o);
    const float16 ttop = invR * (box - r_o);

    // re-order intersections to find smallest and largest on each axis
    const float16 tmin = min(ttop, tbot);
    const float16 tmax = max(ttop, tbot);

    // find the largest tmin and the smallest tmax
    const float4 largest_tmin  = max(max(tmin.s048c, tmin.s159d), tmin.s26ae);
    const float4 smallest_tmax = min(min(tmax.s048c, tmax.s159d), tmax.s26ae);

	*tnear = largest_tmin;
	*tfar = smallest_tmax;
}

float4 backProject(__global float* m, float4 v)
{
    return (float4)(
        dot(v, (float4)(m[0], m[4], m[ 8], m[12])),
        dot(v, (float4)(m[1], m[5], m[ 9], m[13])),
        dot(v, (float4)(m[2], m[6], m[10], m[14])),
        1.0f
        );
}

float3 getGradient(__global float4 *data, const uint4 size, const uint4 offset, const uint4 index)
{
    const float4 indexf = convert_float4(index);
    const int4 previous = convert_int4(max(indexf - 1.0f, (float4)(0.0f)));
    const int4 next = convert_int4(min(indexf + 1.0f, convert_float4(size - 1)));

    float3 gradient = (float3)(
        data[ next.x + index.y * offset.y + index.z * offset.z].w - data[previous.x +    index.y * offset.y +    index.z * offset.z].w,
        data[index.x +  next.y * offset.y + index.z * offset.z].w - data[   index.x + previous.y * offset.y +    index.z * offset.z].w,
        data[index.x + index.y * offset.y +  next.z * offset.z].w - data[   index.x +    index.y * offset.y + previous.z * offset.z].w
        );

    return normalize(gradient);
}

void phongShading(float16 *color, const float16 gradient, const float16 bisector, const float4 light, const float4 lightDiffuse, const float4 lightSpecular, const float4 lightAmbient, const float materialShininess)
{
    *color = lightAmbient.xyzwxyzwxyzwxyzw * *color;
    *color += lightDiffuse.xyzwxyzwxyzwxyzw * (float16)(
        lightDiffuse * max(dot(gradient.s0123, light), 0.0f),
        lightDiffuse * max(dot(gradient.s4567, light), 0.0f),
        lightDiffuse * max(dot(gradient.s89ab, light), 0.0f),
        lightDiffuse * max(dot(gradient.scdef, light), 0.0f)
        );
    *color += lightSpecular.xyzwxyzwxyzwxyzw * (float16)(
        lightSpecular * pow(max(dot(gradient.s0123, bisector.s0123), 0.0), materialShininess),
        lightSpecular * pow(max(dot(gradient.s4567, bisector.s4567), 0.0), materialShininess),
        lightSpecular * pow(max(dot(gradient.s89ab, bisector.s89ab), 0.0), materialShininess),
        lightSpecular * pow(max(dot(gradient.scdef, bisector.scdef), 0.0), materialShininess)
        );
}

__kernel void volumeRendering(
    __global float *inverseViewMatrix,
    __global float4 *colorBuffer,
    __read_only __global float4 *volumeData, const uint4 volumeSize,
    __read_only image2d_t transferFunction, sampler_t transferFunctionSampler,
    const uint imageWidth, const uint imageHeight,
    const float volumeOffset, const float volumeScale, const float volumeStepSize, const float4 volumeThickness, const uint histogramSize,
    const float4 lightDiffuse, const float4 lightSpecular, const float4 lightAmbient, const float4 lightDirection, const float materialShininess,
    __global uint *bufferData
#ifdef __CL_ENABLE_DEBUG
    , const uint debugSize
    , __global float *debugData
#endif
        )
{
    const uint x = get_global_id(0) * 2;
    const uint y = get_global_id(1) * 2;
    if (x >= imageWidth || y >= imageHeight) return;

#ifdef __CL_ENABLE_DEBUG
    if (x + y * imageWidth < debugSize)
        debugData[x + y * imageWidth] = 0.0f;
#endif
    const uint imageSize = min(imageWidth, imageHeight);
    const float imageScale = imageSize == 1 ? 1.0f : 1.0f / (imageSize - 1);
    const float8 imageCoords = (float8)(
        ((x + 0.5f) * 2 - imageWidth + 1) * imageScale, ((y + 0.5f) * 2 - imageHeight + 1) * imageScale,
        ((x + 1.5f) * 2 - imageWidth + 1) * imageScale, ((y + 0.5f) * 2 - imageHeight + 1) * imageScale,
        ((x + 0.5f) * 2 - imageWidth + 1) * imageScale, ((y + 1.5f) * 2 - imageHeight + 1) * imageScale,
        ((x + 1.5f) * 2 - imageWidth + 1) * imageScale, ((y + 1.5f) * 2 - imageHeight + 1) * imageScale
        );

    // calculate eye ray in world space
    const float depth = -4.0f;
    const float4 pinhole = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 origin  = (float4)(backProject(inverseViewMatrix, pinhole).xyz, 0.0f);
    const float16 rayOrigin = origin.xyzwxyzwxyzwxyzw;
    const float16 rayDirection = (float16)(
        backProject(inverseViewMatrix, normalize((float4)(imageCoords.s01, depth, 0.0f))).xyz, 0.0f,
        backProject(inverseViewMatrix, normalize((float4)(imageCoords.s23, depth, 0.0f))).xyz, 0.0f,
        backProject(inverseViewMatrix, normalize((float4)(imageCoords.s45, depth, 0.0f))).xyz, 0.0f,
        backProject(inverseViewMatrix, normalize((float4)(imageCoords.s67, depth, 0.0f))).xyz, 0.0f
        );
    
    // find intersection with box
	float4 tnear = (float4)(0.0f);
    float4 tfar  = (float4)(0.0f);
    float16 boxSize = volumeThickness.xyzwxyzwxyzwxyzw;
    intersectRay(rayOrigin , rayDirection, boxSize, &tnear, &tfar);

    int4 valid = isless(tnear, tfar);
    if (!any(valid))
    {
        // write output color
        colorBuffer[(x + 0) + (y + 0) * imageWidth] = (float4)(0.0f);
        colorBuffer[(x + 1) + (y + 0) * imageWidth] = (float4)(0.0f);
        colorBuffer[(x + 0) + (y + 1) * imageWidth] = (float4)(0.0f);
        colorBuffer[(x + 1) + (y + 1) * imageWidth] = (float4)(0.0f);
        return;
    }
    tnear = max(tnear, (float4)(0.0f));  // clamp to near plane

    uint size = histogramSize * 2;
    uint groupOffset = get_group_id(0) + get_group_id(1) * get_num_groups(0);
    __global uint *data = bufferData + groupOffset * size;
    
    // march along ray from back to front, accumulating color
    const uint4 offset = (uint4)(1, volumeSize.x, volumeSize.x * volumeSize.y, 0);
    const float4  sampleOffset = convert_float4(volumeSize - 1) * 0.5;
    const float16 sampleScale = sampleOffset.xyzwxyzwxyzwxyzw / boxSize;
    const float16 sampleOrigin = rayOrigin * sampleScale + sampleOffset.xyzwxyzwxyzwxyzw;
    const float16 sampleDirection = rayDirection * sampleScale;
    const float16 angleBisector  = (float16)(
        normalize(lightDirection - rayDirection.s0123),
        normalize(lightDirection - rayDirection.s4567),
        normalize(lightDirection - rayDirection.s89ab),
        normalize(lightDirection - rayDirection.scdef)
        );
    
    float16 sampleColor[GAUSSIAN_1D];
        for (int i = 0; i < GAUSSIAN_1D; i++) sampleColor[i] = (float16)(0.0f);
    float16 sampleValue = (float16)(0.0f);
    float16 pixelColor = (float16)(0.0f);
    float8 scalar = (float8)(0.5f);
    float transparent = 1.0f;
    for(float4 t = tnear; any(valid = isless(t, tfar)); t += (float4)(volumeStepSize))
    {
        // read from 3D texture
        float16 samplePosition = sampleOrigin + sampleDirection * t.xxxxyyyyzzzzwwww;
        uint16 sampleNearest = convert_uint16(samplePosition);
        uint4 sampleValid = select((uint4)(0), (uint4)(1), valid);
        uint4 sampleIndex = sampleValid * (sampleNearest.s048c + sampleNearest.s159d * offset.y + sampleNearest.s26ae * offset.z);
        float16 sampleValue = (float16)(volumeData[sampleIndex.x], volumeData[sampleIndex.y], volumeData[sampleIndex.z], volumeData[sampleIndex.w]);
        scalar.even = clamp((sampleValue.s37bf + volumeOffset) * volumeScale, 0.0f, 1.0f);
        
#ifdef __CL_ENABLE_COMPUTING_ENTROPY
        // compute view entropy
        uint v = (uint)((scalar.s0 + scalar.s2 + scalar.s4 + scalar.s6) * 0.25f * (histogramSize - 1));
        ulong visibility = (ulong)(transparent * 1000.0f + 0.5f);
        ulong entropy = transparent == 0.0f ? 0 : (ulong)(-transparent * log2(transparent) * 1000.0f + 0.5f);
        ulong final = entropy << 32 | visibility;
        if (final > 0) atom_add((__global ulong*)data + v, final);
#endif
        for (int i = GAUSSIAN_1D - 1; i > 0; i--)
            sampleColor[i] = sampleColor[i - 1];

        // lookup in transfer function texture
        sampleColor[0] = convert_float16(sampleValid.xxxxyyyyzzzzwwww) * (float16)(
            read_imagef(transferFunction, transferFunctionSampler, scalar.s01),
            read_imagef(transferFunction, transferFunctionSampler, scalar.s23),
            read_imagef(transferFunction, transferFunctionSampler, scalar.s45),
            read_imagef(transferFunction, transferFunctionSampler, scalar.s67)
            );
        transparent *= 1.0f - (sampleColor[0].s3 + sampleColor[0].s7 + sampleColor[0].sb + sampleColor[0].sf) * 0.25;
        
#ifdef __CL_ENABLE_SHADING
        // use illumination model
        phongShading(sampleColor, sampleValue, angleBisector, lightDirection, lightDiffuse, lightSpecular, lightAmbient, materialShininess);
#endif
        float16 color = (float16)(0.0f);
        for (int i = 0; i < GAUSSIAN_1D; i++)
            color += sampleColor[i] * gaussian1[i];
        
        // accumulate result
        float4 alpha = color.s37bf * ((float4)(1.0f) - pixelColor.s37bf);
        pixelColor += (float16)(color.s012, 1.0f, color.s456, 1.0f, color.s89a, 1.0f, color.scde, 1.0f) * alpha.xxxxyyyyzzzzwwww;
        
        // early ray termination
        if (!any(isless(pixelColor.s37bf, (float4)(1.0f)))) break;
    }
    
    // march along ray from back to front, accumulating color
    colorBuffer[(x + 0) + (y + 0) * imageWidth] = pixelColor.s0123;
    colorBuffer[(x + 1) + (y + 0) * imageWidth] = pixelColor.s4567;
    colorBuffer[(x + 0) + (y + 1) * imageWidth] = pixelColor.s89ab;
    colorBuffer[(x + 1) + (y + 1) * imageWidth] = pixelColor.scdef;
    
#ifdef __CL_ENABLE_DEBUG
    if (x + y * imageWidth < debugSize)
        debugData[x + y * imageWidth] = (color.x + color.y + color.z) / 3;
#endif
}