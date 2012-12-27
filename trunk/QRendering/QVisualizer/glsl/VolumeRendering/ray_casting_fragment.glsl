/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    ray_casting_fragment.glsl
 * @brief   fragment shader for ray casting.
 * 
 * This file ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/20
 */

#define EPSILON     1e-5
#define FLT_MAX     3.402823466e+38

uniform sampler2DRect prevDepth;
uniform sampler2DRect currDepth;
uniform sampler2DRect currFrontBlender;
uniform sampler3D volumeData;
uniform sampler1D transferFunctionData;
uniform vec3 modelScale;
uniform vec3 sampleScale;
uniform float volumeOffset, volumeScale;
uniform float stepSize;

// Define the boundaries for early ray termination
vec3 minPosition = vec3(0.0);
vec3 maxPosition = vec3(1.0);

// Define the maximum number of iterations for raycasting
int maxIterations = 65536;

vec4 rayCasting(float prevDepth, float currDepth)
{
    // Initialize destination color
    vec4 color = vec4(0.0);
    
    // Compute the depth of the fragment
    float fragDepth = gl_FragCoord.z / gl_FragCoord.w;

    float depth = fragDepth;
    if (depth <= prevDepth) depth = prevDepth;
    if (depth >= currDepth) return color;

    // Compute the camera position by translating the origin to the center of the volumeData
    vec3 cameraPosition = gl_ModelViewMatrixInverse[3].xyz; // for gl version 130
    
    // Compute the ray direction
    vec3 rayDirection = normalize(gl_TexCoord[0].xyz * modelScale - cameraPosition) * sampleScale;
    vec3 rayStart = gl_TexCoord[0].xyz - rayDirection * fragDepth;

    // Move the sampling postion to the next grid point
    depth += stepSize - mod(depth, stepSize);

    // Initialize number of iterations
    int count = 0;
    
    // Ray-casting algorithm
    while (depth <= currDepth)
    {
        vec3 rayPosition = rayStart + rayDirection * depth;

        // Terminate loop if outside volumeData
        if (any(greaterThan(rayPosition, maxPosition))) break;
        
        if (any(lessThan(rayPosition, minPosition))) break;
        
        // Lookup new scalar value
        float scalar = (texture(volumeData, rayPosition).r + volumeOffset) * volumeScale;
        
        // Do color integration
        vec4 integration = texture(transferFunctionData, scalar); // turn off pre-integration
        
        // Perform blending
        color += vec4(integration.rgb, 1.0) * integration.a * (1.0 - color.a); // turn off phong model

        // Early ray termination
        if (color.a >= 1.0) break;
        
        // Terminate if too many iterations
        if (++count >= maxIterations) break;
        
        // Move one step forward
        depth += stepSize;
    }

    return color;
}

void main()
{
    //
	// window-space depth interpolated linearly in screen space
	float fragDepth = gl_FragCoord.z;
    
	vec2 prevDepth = texture(prevDepth, gl_FragCoord.xy).xy;
    vec2 currDepth = texture(currDepth, gl_FragCoord.xy).xy;
    vec4 currFront = texture(currFrontBlender, gl_FragCoord.xy);
    
	float prevNearestDepth = -prevDepth.x;
	float prevFarthestDepth = prevDepth.y;
	float currNearestDepth = -currDepth.x;
	float currFarthestDepth = currDepth.y;

    if (currNearestDepth == FLT_MAX && currFarthestDepth == -FLT_MAX)
    {
        if (prevNearestDepth == FLT_MAX && prevFarthestDepth == -FLT_MAX)
        {
            gl_FragData[0] = vec4(0.0);
            gl_FragData[1] = vec4(0.0);
            return;
        }
        else if (prevNearestDepth == -FLT_MAX && prevFarthestDepth == FLT_MAX)
        {
            gl_FragData[0] = rayCasting(-FLT_MAX, FLT_MAX);
            gl_FragData[1] = vec4(0.0);
            return;
        }
    }

    gl_FragData[0] = rayCasting(currFarthestDepth, prevFarthestDepth);

    vec4 color = rayCasting(prevNearestDepth, currNearestDepth);
    gl_FragData[1] = currFront + vec4(color.rgb, 1.0) * color.a * (1.0 - currFront.a);
    //
    //gl_FragColor = rayCasting(0.0, FLT_MAX);
}