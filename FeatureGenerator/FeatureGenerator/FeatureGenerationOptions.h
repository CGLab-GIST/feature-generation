
//  Copyright(c) 2018 CGLab, GIST. All rights reserved.
//  
//  Redistribution and use in source and binary forms, with or without modification, 
//  are permitted provided that the following conditions are met :
// 
//  - Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation
//    and / or other materials provided with the distribution.
//  - Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
//  
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//  ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
//  DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
//  OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#pragma once

#define USE_BC_BAGGING				// Bagging
#define USE_VISIBILITY_HEURISTIC	// Ambient light
#define USE_PREFILTER				// use prefiltering for new feature


#ifndef USE_BC_BAGGING
#ifdef USE_PREFILTER				// prefiltering is only possible when bagging turns on due to variance of our new feature
#undef USE_PREFILTER
#endif
#endif


// parameters for bagging
#define NUM_BIN 8
#define MAX_ITER_BAGGING 10
#define AMBIENT_TERM 0.01

#include <vector_types.h>
#include <stdlib.h>
#include <memory.h>
#include <algorithm>

// BAGG_PIXEL
struct BAGG_PIXEL {
	float4 m_accX2[NUM_BIN];
	float4 m_accX[NUM_BIN];

	BAGG_PIXEL() {
		init();
	}
	void init() {
		memset(m_accX2, 0, sizeof(float4) * NUM_BIN);
		memset(m_accX, 0, sizeof(float4) * NUM_BIN);
	}
	inline void add(float4& dest, float r, float g, float b, float w) {
		dest.x += r;
		dest.y += g;
		dest.z += b;
		dest.w += w;
	}
	inline void add(float4& dest, const float4& src) {
		dest.x += src.x;
		dest.y += src.y;
		dest.z += src.z;
		dest.w += src.w;
	}
	inline void add(float r, float g, float b, float w, int spp) {
		int i = spp % NUM_BIN;
		add(m_accX[i], r, g, b, w);
		add(m_accX2[i], r * r, g * g, b * b, 0.f);
	}
	inline void accumulate(const BAGG_PIXEL& pixel) {
		for (int i = 0; i < NUM_BIN; ++i) {
			add(m_accX[i], pixel.m_accX[i]);
			add(m_accX2[i], pixel.m_accX2[i]);
		}
	}
};

struct BAGG_VAR {
	float3 m_accX;
	float3 m_accX2;
	BAGG_VAR() {
		init();
	}
	inline void init() {
		m_accX.x = m_accX.y = m_accX.z = 0.f;
		m_accX2.x = m_accX2.y = m_accX2.z = 0.f;
	}
	inline void add(float3& dest, const float3& src) {
		dest.x += src.x;
		dest.y += src.y;
		dest.z += src.z;
	}
	inline void add(float3& dest, float r, float g, float b) {
		dest.x += r;
		dest.y += g;
		dest.z += b;
	}
	inline void add(float r, float g, float b, float w) {
		add(m_accX, r, g, b);
		add(m_accX2, r * r, g * g, b * b);
	}
	inline void add(const BAGG_VAR& _var) {
		add(m_accX, _var.m_accX);
		add(m_accX2, _var.m_accX2);
	}
	inline void add(const float3& _accX, const float3& _accX2) {
		add(m_accX, _accX);
		add(m_accX2, _accX2);
	}

	// Variance of Sample Mean
	inline float4 calcVarMean(const float sumW) {
		float invW = 1.f / fmaxf(1e-4f, sumW);		
		float4 varMean;
		varMean.x = std::max(0.f, invW * (m_accX2.x * invW - (m_accX.x * invW) * (m_accX.x * invW)));
		varMean.y = std::max(0.f, invW * (m_accX2.y * invW - (m_accX.y * invW) * (m_accX.y * invW)));
		varMean.z = std::max(0.f, invW * (m_accX2.z * invW - (m_accX.z * invW) * (m_accX.z * invW)));
		varMean.w = 0.f;		
		return varMean;
	}
};

