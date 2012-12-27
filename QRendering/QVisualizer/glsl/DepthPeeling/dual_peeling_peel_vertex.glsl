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
	gl_Position = ftransform();
	gl_TexCoord[0].w = abs(normalize(gl_NormalMatrix * gl_Normal).z);
}
