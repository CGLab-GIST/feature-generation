
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

#include "FeatureGenerationOptions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <vector_types.h>
#include <vector>

class FeatureGenerator {
private:
	float4 *m_tempCol;
	float4 *m_newFeature, *m_varNewFeature;
	int m_width, m_height, m_nPix;
	
	// pointers 
	const std::vector<float4> *p_img, *p_varImg, *p_veryDirectImg;

public:
	bool m_isInit;

	FeatureGenerator();
	~FeatureGenerator();
	void clearMemory();
	void allocMemory(int xSize, int ySize);

	// Setup function
	void setupImage(std::vector<float4>& _img, std::vector<float4>& _varImg, std::vector<float4>& _veryDirectImg) {
		p_img = &_img;
		p_varImg = &_varImg;
		p_veryDirectImg = &_veryDirectImg;
	}

	// Feature generation function
	void generateNewFeature(std::vector< BAGG_PIXEL >& dxSamples, std::vector< BAGG_PIXEL >& dySamples, std::vector< BAGG_PIXEL >& initSamples, const bool isLastPass);
	

	inline float4* getFeature() { return m_newFeature; }
	inline float4* getVarFeature() { return m_varNewFeature; }
};
