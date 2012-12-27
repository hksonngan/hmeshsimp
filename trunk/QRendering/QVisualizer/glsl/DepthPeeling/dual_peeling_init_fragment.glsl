//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

void main()
{
    float fragDepth = gl_FragCoord.z;
    float fragDistance = fragDepth / gl_FragCoord.w;

	gl_FragData[0].xy = vec2(-fragDepth, fragDepth);
    gl_FragData[1].xy = vec2(-fragDistance, fragDistance);
}
