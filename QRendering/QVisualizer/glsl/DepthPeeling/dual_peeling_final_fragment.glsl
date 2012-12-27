//--------------------------------------------------------------------------------------
// Order Independent Transparency with Dual Depth Peeling
//
// Author: Louis Bavoil
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

uniform sampler2DRect frontBlender;
uniform sampler2DRect backBlender;

void main()
{
	vec4 frontColor = texture(frontBlender, gl_FragCoord.xy);
	vec4 backColor = texture(backBlender, gl_FragCoord.xy);

	// front + back
	gl_FragColor = frontColor + vec4(backColor.rgb, 1.0) * backColor.a * (1.0 - frontColor.a);;

	// front blender
	//gl_FragColor = frontColor;
	
	// back blender
	//gl_FragColor = backColor;
}
