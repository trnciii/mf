#pragma once

#include "vector_math.h"


#ifdef __CUDACC__

#define __device__ __device__

#include <curand_kernel.h>

struct RNG{

	__device__ RNG(int i=0){
		curand_init(0, i, 0, &state);
	}

	__device__ float uniform(){
		return curand_uniform(&state);
	}

	__device__ float2 uniform2(){
		return make_float2(curand_uniform(&state), curand_uniform(&state));
	}

	__device__ float3 uniform3(){
		return make_float3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state));
	}

private:
	curandState state;
};

#else // __CUDACC__

#define __device__

#include <random>

struct RNG{

	RNG(uint32_t i=0):state(i), dist(0.0, 1.0){}

	float uniform(){
		return dist(state);
	}

	float2 uniform2(){
		return make_float2(dist(state), dist(state));
	}

	float3 uniform3(){
		return make_float3(dist(state), dist(state), dist(state));
	}

private:
	std::mt19937 state;
	std::uniform_real_distribution<> dist;
};

#endif // __CUDACC__



__device__ float3 sample_uniform_hemisphere(float u1, float u2){
	float r = sqrt(1 - u1*u1);
	u2 *= 2*M_PI;
	return make_float3(r*cos(u2), r*sin(u2), u1);
}



namespace GGX{

__device__ float NDF(float3 m, float alpha){
	if(m.z <= 0) return 0;
	if(alpha < 1e-6) return (fabs(1 - m.z) < 1e-6)? 1 : 0;

	float a2 = alpha * alpha;
	float cos2 = m.z * m.z;
	float tan2 = 1/(cos2) -1;
	return a2 / (M_PI * cos2*cos2 * (a2 + tan2)*(a2 + tan2));
}


__device__ float G1_distant(float3 v, float alpha){
	float t = 1/(v.z*v.z) - 1;
	return 2/(1 + sqrt(1 + alpha*alpha*t*t));
}

__device__ float G1(float3 v, float3 m, float alpha){
	if(v.z > 1-1e-6) return 1;
	if(dot(v, m)/v.z < 0) return 0;

	float t = 1/(v.z*v.z) - 1;
	return 2/(1 + sqrt(1 + alpha*alpha*t*t));
}


__device__ float normalization_constraint(float3 v, float alpha, uint32_t sample_size){
	float accum = 0;
	RNG rng(0);

	for(int i=0; i<sample_size; i++){
		float3 m = sample_uniform_hemisphere(rng.uniform(), rng.uniform());

		float mv = dot(m, v);
		if(mv <= 0) continue;

		accum += mv * NDF(m, alpha);
	}

	accum *= G1_distant(v, alpha)/sample_size*2*M_PI;

	return accum;
}


} // GGX
