
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

#include <random>
#include <time.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h> 
#include <thrust/functional.h> 
#include <thrust/device_vector.h> 
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <curand_kernel.h>

#include "PoissonSolver.h"

template <class T> __device__ __forceinline__ T ld4(const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 4) *(float*)((char*)&out + ofs) = __ldg((float*)((char*)&in + ofs)); return out; }
template <class T> __device__ __forceinline__ T ld8(const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 8) *(float2*)((char*)&out + ofs) = __ldg((float2*)((char*)&in + ofs)); return out; }
template <class T> __device__ __forceinline__ T ld16(const T& in)   { T out; for (int ofs = 0; ofs < sizeof(T); ofs += 16) *(float4*)((char*)&out + ofs) = __ldg((float4*)((char*)&in + ofs)); return out; }

// Texture memory for pre-filter
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_img, g_varImg;

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define SOLVER_EP 1e-5f

inline int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

template <typename T>
struct square
{
	__host__ __device__
		float4 operator()(const T& x) const {
		return x * x;
	}
};

class SolverGPUMem {
public:
	float4* _e;
	float4* _b;
	float4* _r;
	float4* _p;
	float4* _x;
	float4* _Ap;
	float*  _w2;

	float4* _xiAxi;

	BAGG_PIXEL* _rawDx;
	BAGG_PIXEL* _rawDy;
	BAGG_PIXEL* _rawInit;
	float4* _feat;
	float4* _varFeat;

	SolverGPUMem(int nPix) {
		checkCudaErrors(cudaMalloc((void **)&_e, 2 * nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_b, 2 * nPix * sizeof(float4)));		
		checkCudaErrors(cudaMalloc((void **)&_r,  nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_p,  nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_x,  nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_Ap, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_w2, 2 * nPix * sizeof(float)));	

		checkCudaErrors(cudaMalloc((void **)&_xiAxi, nPix * sizeof(float4)));

		checkCudaErrors(cudaMalloc((void **)&_rawDx, nPix * sizeof(BAGG_PIXEL)));
		checkCudaErrors(cudaMalloc((void **)&_rawDy, nPix * sizeof(BAGG_PIXEL)));
		checkCudaErrors(cudaMalloc((void **)&_rawInit, nPix * sizeof(BAGG_PIXEL)));
		checkCudaErrors(cudaMalloc((void **)&_feat, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_varFeat, nPix * sizeof(float4)));
		
		checkCudaErrors(cudaGetLastError());
	}
	~SolverGPUMem() {
		checkCudaErrors(cudaFree(_e));
		checkCudaErrors(cudaFree(_b));
		checkCudaErrors(cudaFree(_r));
		checkCudaErrors(cudaFree(_p));
		checkCudaErrors(cudaFree(_x));
		checkCudaErrors(cudaFree(_Ap));
		checkCudaErrors(cudaFree(_w2));

		checkCudaErrors(cudaFree(_xiAxi));		

		checkCudaErrors(cudaFree(_rawDx));
		checkCudaErrors(cudaFree(_rawDy));
		checkCudaErrors(cudaFree(_rawInit));
		checkCudaErrors(cudaFree(_feat));
		checkCudaErrors(cudaFree(_varFeat));

		checkCudaErrors(cudaGetLastError());
	}
};

__device__ float getWeightFromVariance(const float4& var) {
	return (1.f / (fmaxf(0.f, var.x + var.y + var.z) + 1e-4f));
}

__device__ float getWeightFromVariance(const float4& varDx, const float4& varDy) {
	float wDx = getWeightFromVariance(varDx);
	float wDy = getWeightFromVariance(varDy);
	float w = (wDx + wDy) / 2.f;
	return w;
}

__global__ void kernel_calc_Px(float4* _e, const float4* _x, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;

	_e[i] = (cx != xSize - 1) ? _x[i + 1] - _x[i] : make_float4(0.f);
	_e[xSize * ySize + i] = (cy != ySize - 1) ? _x[i + xSize] - _x[i] : make_float4(0.f);
}

__global__ void kernel_calc_axpy(float4* _e, const float4* _b, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_e[2 * i] = _b[2 * i] - _e[2 * i];
	_e[2 * i + 1] = _b[2 * i + 1] - _e[2 * i + 1];
}

__global__ void kernel_calc_PTW2x_(float4* _r, const float* _w2, const float4* _e, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	const int n = xSize * ySize;

	float4 PTW2xi = make_float4(0.f);
	if (cx != 0)
		PTW2xi += _w2[i - 1] * _e[i - 1];
	if (cx != xSize - 1)
		PTW2xi -= _w2[i] * _e[i];
	if (cy != 0)
		PTW2xi += _w2[n + i - xSize] * _e[n + i - xSize];
	if (cy != ySize - 1)
		PTW2xi -= _w2[n + i] * _e[n + i];
	_r[i] = PTW2xi;
}

__global__ void kernel_calc_Ax_xAx(float4* _Ap, float4* _xiAxi, const float* _w2, const float4* _p, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	const int n = xSize * ySize;

	float4 xi = ld16(_p[i]);
	float4 Axi = make_float4(0.f);
	if (cx != 0)            
		Axi += ld4(_w2[i - 1]) * (xi - ld16(_p[i - 1]));
	if (cx != xSize - 1)  
		Axi += ld4(_w2[i]) * (xi - ld16(_p[i + 1]));
	if (cy != 0)            
		Axi += ld4(_w2[n + i - xSize]) * (xi - ld16(_p[i - xSize]));
	if (cy != ySize - 1) 
		Axi += ld4(_w2[n + i]) * (xi - ld16(_p[i + xSize]));
	_Ap[i] = Axi;
	_xiAxi[i] = xi * Axi;
}

__global__ void kernel_calc_r_rz(float4* _r, const float4* _Ap, const float4 a, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_r[i] = _r[i] - _Ap[i] * a;
}

__global__ void kernel_update_p_(float4* _p, const float4* _r, const float4 b, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;	
	_p[i] = _r[i] + b * _p[i];
}

__global__ void kernel_update_x_(float4* _x, const float4* _p, const float4 a, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_x[i] += _p[i] * a;
}

__global__ void kernel_scale_weights(float* _x, const float scalar, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_x[i] *= scalar;
	_x[i + xSize * ySize] *= scalar;
}

__global__ void kernel_buffer_divide(float4* _x, float* _weight, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_x[i] /= fmaxf(SOLVER_EP, _weight[i]);	
}

__global__ void kernel_update_mean_variance(float4* _mean, float4* _M2, float* _sumWeight, const float4* _x, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;

	// uniform weighting
	float weight = 1.f;

	// Online mean & variance computation
	const float4& data = _x[i];
	float wSum = _sumWeight[i] + weight;
	float4 meanOld = _mean[i];
	float4 meanNew = meanOld + (weight / wSum) * (data - meanOld);
	_M2[i] += weight * (data - meanOld) * (data - meanNew);
	_mean[i] = meanNew;
	_sumWeight[i] = wSum;
}

__global__ void kernel_set_gradients_weight(float4* _b, float* _w, float4* _initSol, const BAGG_PIXEL* _rawDx, const BAGG_PIXEL* _rawDy, const BAGG_PIXEL* _rawInit, const int* _randIdx, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	int nPix = xSize * ySize;

	float4 accDx = make_float4(0.f);
	float4 accDx2 = make_float4(0.f);
	float4 accDy = make_float4(0.f);
	float4 accDy2 = make_float4(0.f);
	float4 accInit = make_float4(0.f);

	for (int s = 0; s < NUM_BIN; ++s) {
		// Resampling by uniform distribution
#ifdef USE_BC_BAGGING
		int idxX = _randIdx[3 * s + 0];
		int idxY = _randIdx[3 * s + 1];
		int idxInit = _randIdx[3 * s + 2];
#else
		int idxX = s;
		int idxY = s;
		int idxInit = s;
#endif	

		accDx += _rawDx[i].m_accX[idxX];
		accDx2 += _rawDx[i].m_accX2[idxX];
		
		accDy += _rawDy[i].m_accX[idxY];
		accDy2 += _rawDy[i].m_accX2[idxY];

		accInit += _rawInit[i].m_accX[idxInit];		
	}
	float4 meanDx = accDx / accDx.w;		
	float4 meanDy = accDy / accDy.w;
	float4 meanInit = accInit / accInit.w;

	_initSol[i] = meanInit;
	
	float4 varDx = (accDx2 / accDx.w - meanDx * meanDx) / accDx.w;
	float4 varDy = (accDy2 / accDy.w - meanDy * meanDy) / accDy.w;
	
	// A diagonal matrix whose elements are the reciprocals of variances
	_w[0 * nPix + i] = getWeightFromVariance(varDx);
	_w[1 * nPix + i] = getWeightFromVariance(varDy);

	_b[0 * nPix + i] = meanDx;
	_b[1 * nPix + i] = meanDy;		
}

__device__ float calc_feat_weight(float4& dist2, int patchRadius) {
	int nEle = 3 * (2 * patchRadius + 1) * (2 * patchRadius + 1);
	float avgDist2 = fmaxf(0.f, (dist2.x + dist2.y + dist2.z) / (float)nEle);
	return (expf(-avgDist2));
}

__global__ void kernel_prefilter_feature(float4* _outImg, int xSize, int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int cIdx = cy * xSize + cx;
	const int HALF_WINDOW_NLM = 5;
	const float epsilon = 1e-10f;

	int2 sWindow = make_int2(cx - HALF_WINDOW_NLM, cy - HALF_WINDOW_NLM);
	int2 eWindow = make_int2(cx + HALF_WINDOW_NLM, cy + HALF_WINDOW_NLM);

	float sumW = 0.f;
	float4 outCol = make_float4(0.f);
	const int radiusPatch = 3;
	const float kf = 1.f;

	for (int y = sWindow.y; y <= eWindow.y; ++y) {
		for (int x = sWindow.x; x <= eWindow.x; ++x) {
			// patchwise distance for the color
			float4 dist_col = make_float4(0.f);
			for (int py = -radiusPatch; py <= radiusPatch; ++py) {
				for (int px = -radiusPatch; px <= radiusPatch; ++px) {
					const float4& pc_color = tex2D(g_img, cx + px, cy + py);
					const float4& pi_color = tex2D(g_img, x + px, y + py);
					const float4& pc_varColor = tex2D(g_varImg, cx + px, cy + py);
					const float4& pi_varColor = tex2D(g_varImg, x + px, y + py);

					float4 pq_varColor = fminf(pc_varColor, pi_varColor);
					float4 diffCol = (pc_color - pi_color) * (pc_color - pi_color) - (pc_varColor + pq_varColor);
					diffCol /= (make_float4(epsilon) + kf * kf * (pc_varColor + pi_varColor));
					dist_col += diffCol;
				}
			}
			float w = calc_feat_weight(dist_col, radiusPatch);
			const float4& i_img = tex2D(g_img, x, y);
			outCol += w * i_img;
			sumW += w;
		}
	}

	if (sumW > 0.0)
		_outImg[cIdx] = outCol / sumW;
	else
		_outImg[cIdx] = tex2D(g_img, cx, cy);	
}

extern "C"
void run_conjugate_gradient_gpu(SolverGPUMem& gpuMem, int xSize, int ySize, int nCGIteration) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int nPix = xSize * ySize;
	const int blockDim = 16;

	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	square<float4>       unary_op;
	thrust::plus<float4> binary_op;

	float sumWeight = thrust::reduce(thrust::device, gpuMem._w2, gpuMem._w2 + 2 * nPix, 0.f);		
	float scaleWeight = (float)(xSize * ySize * 2) / sumWeight;

	kernel_scale_weights << <grid, threads >> >(gpuMem._w2, scaleWeight, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());
	
	// Px
	kernel_calc_Px << <grid, threads >> >(gpuMem._e, gpuMem._x, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());

	// e = (b - Px)
	kernel_calc_axpy << <grid, threads >> >(gpuMem._e, gpuMem._b, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());

	// r = PtW(b - Px)
	kernel_calc_PTW2x_ << <grid, threads >> >(gpuMem._r, gpuMem._w2, gpuMem._e, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());

	// setup arguments
	thrust::device_ptr<float4> thrust_r = thrust::device_pointer_cast(gpuMem._r);
	thrust::device_ptr<float4> thrust_xiAxi = thrust::device_pointer_cast(gpuMem._xiAxi);

	// p = r
	checkCudaErrors(cudaMemcpy(gpuMem._p, gpuMem._r, nPix * sizeof(float4), cudaMemcpyDeviceToDevice));

	const float tol = 1e-10f;
	float4 rz = make_float4(0.f);
	float4 rz2 = make_float4(0.f);
	
	for (int cgIter = 0; cgIter < nCGIteration; ++cgIter) {
		if (cgIter == 0)
			rz = thrust::transform_reduce(thrust_r, thrust_r + nPix, unary_op, make_float4(0.f), binary_op);
		else
			rz = rz2;

		kernel_calc_Ax_xAx << <grid, threads >> >(gpuMem._Ap, gpuMem._xiAxi, gpuMem._w2, gpuMem._p, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());

		float4 pAp = thrust::reduce(thrust_xiAxi, thrust_xiAxi + nPix, make_float4(0.f), binary_op);

		float4 a = rz / fmaxf(pAp, make_float4(FLT_MIN));

		kernel_update_x_ << <grid, threads >> > (gpuMem._x, gpuMem._p, a, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());

		kernel_calc_r_rz << <grid, threads >> > (gpuMem._r, gpuMem._Ap, a, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());

		rz2 = thrust::transform_reduce(thrust_r, thrust_r + nPix, unary_op, make_float4(0.f), binary_op);

		if (rz2.x + rz2.y + rz2.z < tol)
			break;

		float4 b = rz2 / fmaxf(rz, make_float4(FLT_MIN));

		kernel_update_p_ << <grid, threads >> > (gpuMem._p, gpuMem._r, b, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());
	}
}

extern "C"
void filter_feature(float4* _outImg, float4* _inImg, float4* _varImg, int xSize, int ySize) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	int nPix = xSize * ySize;
	const int blockDim = 16;

	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	cudaArray *g_src_img, *g_src_varImg;
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMallocArray(&g_src_img, &channelDesc, xSize, ySize));
	checkCudaErrors(cudaMallocArray(&g_src_varImg, &channelDesc, xSize, ySize));

	checkCudaErrors(cudaMemcpyToArray(g_src_img, 0, 0, _inImg, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_varImg, 0, 0, _varImg, nPix * sizeof(float4), cudaMemcpyHostToDevice));

	g_img.addressMode[0] = g_img.addressMode[1] = cudaAddressModeMirror;
	g_varImg.addressMode[0] = g_varImg.addressMode[1] = cudaAddressModeMirror;

	checkCudaErrors(cudaBindTextureToArray(&g_img, g_src_img, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_varImg, g_src_varImg, &channelDesc));

	float4* d_outImg;
	checkCudaErrors(cudaMalloc((void **)&d_outImg, nPix * sizeof(float4)));

	// The core of filtering
	kernel_prefilter_feature << <grid, threads >> >(d_outImg, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());
	
	checkCudaErrors(cudaMemcpy(_outImg, d_outImg, nPix * sizeof(float4), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaUnbindTexture(&g_img));
	checkCudaErrors(cudaUnbindTexture(&g_varImg));

	checkCudaErrors(cudaFreeArray(g_src_img));
	checkCudaErrors(cudaFreeArray(g_src_varImg));

	checkCudaErrors(cudaFree(d_outImg));

	checkCudaErrors(cudaGetLastError());
}

extern "C"
void solver_poisson_bagging_CI(float4* _outImg, float4* _outVar, const std::vector<BAGG_PIXEL>& _dxSamples, const std::vector<BAGG_PIXEL>& _dySamples, 
							   const std::vector<BAGG_PIXEL>& _initSamples, int xSize, int ySize, const bool isLastPass) {

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int nPix = xSize * ySize;
	const int blockDim = 16;

	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	SolverGPUMem gpuMem = SolverGPUMem(nPix);
	checkCudaErrors(cudaMemcpy(gpuMem._rawDx, &_dxSamples[0], nPix * sizeof(BAGG_PIXEL), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuMem._rawDy, &_dySamples[0], nPix * sizeof(BAGG_PIXEL), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuMem._rawInit, &_initSamples[0], nPix * sizeof(BAGG_PIXEL), cudaMemcpyHostToDevice));

	int nCGIteration = isLastPass ? 50 : 30;
	int MAX_ITER_BAG = 1;

#ifdef USE_BC_BAGGING
	MAX_ITER_BAG = MAX_ITER_BAGGING;
#endif

	// Uniform distribution for resampling of bagging process
	std::mt19937 gen(4569803);
	std::uniform_int_distribution<> dis(0, INT32_MAX);

	const int totalRandNumber = 3 * NUM_BIN * MAX_ITER_BAGGING;

	int _host_rand_idx[totalRandNumber];
	int* _d_rand_idx;
	for (int i = 0; i < totalRandNumber; ++i) {
		_host_rand_idx[i] = dis(gen) % NUM_BIN;
	}


	float* _d_sumWeight;
	checkCudaErrors(cudaMalloc((void **)&_d_rand_idx, totalRandNumber * sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_rand_idx, _host_rand_idx, totalRandNumber * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset((void *)gpuMem._feat, 0, nPix * sizeof(float4)));
	checkCudaErrors(cudaMemset((void *)gpuMem._varFeat, 0, nPix * sizeof(float4)));

	checkCudaErrors(cudaMalloc((void **)&_d_sumWeight, nPix * sizeof(float)));
	checkCudaErrors(cudaMemset((void *)_d_sumWeight, 0, nPix * sizeof(float)));
	
	for (int iterBag = 0; iterBag < MAX_ITER_BAG; ++iterBag) {		
		// Step 1. calc dx & dy & weights
		kernel_set_gradients_weight << <grid, threads >> >(gpuMem._b, gpuMem._w2, gpuMem._x, gpuMem._rawDx, gpuMem._rawDy, gpuMem._rawInit, &_d_rand_idx[iterBag * 3 * NUM_BIN], xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());
				
		// Step 2. run conjugate gradient
		run_conjugate_gradient_gpu(gpuMem, xSize, ySize, nCGIteration);

		// Step 3. accumulate solution
		kernel_update_mean_variance << <grid, threads >> > (gpuMem._feat, gpuMem._varFeat, _d_sumWeight, gpuMem._x, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	// Step 4. variance buffer complete
	kernel_buffer_divide << <grid, threads >> > (gpuMem._varFeat, _d_sumWeight, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());

	// copy results
	checkCudaErrors(cudaMemcpy(_outImg, gpuMem._feat, nPix * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_outVar, gpuMem._varFeat, nPix * sizeof(float4), cudaMemcpyDeviceToHost));
		
	checkCudaErrors(cudaFree(_d_sumWeight));
	checkCudaErrors(cudaFree(_d_rand_idx));	
	checkCudaErrors(cudaGetLastError());
}


 





