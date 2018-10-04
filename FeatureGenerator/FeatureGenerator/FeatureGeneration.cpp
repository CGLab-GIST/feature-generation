
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


#include "FeatureGeneration.h"
#include "FeatureGenerationOptions.h"
#include "PoissonSolver.h"

FeatureGenerator::FeatureGenerator() {
	m_newFeature = m_varNewFeature = NULL;
	m_tempCol = NULL;

	m_isInit = false;
}

void FeatureGenerator::allocMemory(int xSize, int ySize) {
	m_width = xSize;
	m_height = ySize;
	m_nPix = xSize * ySize;

	clearMemory();

	m_newFeature = new float4[m_nPix];
	m_varNewFeature = new float4[m_nPix];
	m_tempCol = new float4[m_nPix];

	m_isInit = true;
}

void FeatureGenerator::clearMemory() {
	if (m_newFeature)		delete[] m_newFeature;
	if (m_varNewFeature)	delete[] m_varNewFeature;
	if (m_tempCol)		delete[] m_tempCol;
}

FeatureGenerator::~FeatureGenerator() {
	clearMemory();
}

void FeatureGenerator::generateNewFeature(std::vector< BAGG_PIXEL >& dxSamples, std::vector< BAGG_PIXEL >& dySamples, std::vector< BAGG_PIXEL >& initSamples, const bool isLastPass) {
	// main feature generation
	//  - 'isLastPass' : It is for an adaptive sampling
	solver_poisson_bagging_CI(m_newFeature, m_varNewFeature, dxSamples, dySamples, initSamples, m_width, m_height, isLastPass);

	// add very direct image
	for (int i = 0; i < m_nPix; i++) {
		m_newFeature[i].x += (*p_veryDirectImg)[i].x;
		m_newFeature[i].y += (*p_veryDirectImg)[i].y;
		m_newFeature[i].z += (*p_veryDirectImg)[i].z;
	}

	// (optional) prefiltering
#ifdef USE_PREFILTER
	filter_feature(m_tempCol, m_newFeature, m_varNewFeature, m_width, m_height);
	memcpy(m_newFeature, m_tempCol, sizeof(float4) * m_nPix);
#endif
}