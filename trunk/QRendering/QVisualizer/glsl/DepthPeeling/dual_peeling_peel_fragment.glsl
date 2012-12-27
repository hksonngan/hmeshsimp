//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#extension GL_ARB_draw_buffers : require

uniform sampler2DRect depthBlender;
uniform sampler2DRect frontBlender;
uniform vec4 Color;

#define MAX_DEPTH   1.0
#define FLT_MAX     3.402823466e+38

void main()
{
	// window-space depth interpolated linearly in screen space
	float fragDepth = gl_FragCoord.z;

	vec2 depth = texture(depthBlender, gl_FragCoord.xy).xy;
	vec4 forwardTemp = texture(frontBlender, gl_FragCoord.xy);
	
	// Front colors always increase (DST += SRC*ALPHA_MULT)
	// so we can use pass-through by default with MAX blending
	gl_FragData[2] = forwardTemp;
    gl_FragData[4] = gl_FragData[2];
	
	// Because over blending makes color increase or decrease,
	// we cannot pass-through by default.
	// Each pass, only one fragment writes a color greater than 0
	gl_FragData[3] = vec4(0.0);

	float nearestDepth = -depth.x;
	float farthestDepth = depth.y;

	if (fragDepth < nearestDepth || fragDepth > farthestDepth) {
		// Skip this depth in the peeling algorithm
		gl_FragData[0].xy = vec2(-MAX_DEPTH);
        gl_FragData[1].xy = vec2(-FLT_MAX);
		return;
	}
	
	if (fragDepth > nearestDepth && fragDepth < farthestDepth) {
        float fragDistance = fragDepth / gl_FragCoord.w;
		// This fragment needs to be peeled again
		gl_FragData[0].xy = vec2(-fragDepth, fragDepth);
        gl_FragData[1].xy = vec2(-fragDistance, fragDistance);
		return;
	}
	
	// If we made it here, this fragment is on the peeled layer from last pass
	// therefore, we need to shade it, and make sure it is not peeled any farther
	vec4 color = vec4(Color.rgb * gl_TexCoord[0].w, Color.a);
	gl_FragData[0].xy = vec2(-MAX_DEPTH);
    gl_FragData[1].xy = vec2(-FLT_MAX);
	
	if (fragDepth == nearestDepth) {
		gl_FragData[2] = forwardTemp + vec4(color.rgb, 1.0) * color.a * (1.0 - forwardTemp.a);
        gl_FragData[4] = gl_FragData[2];
	} else {
		gl_FragData[3] = color;
	}
}
