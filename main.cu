#include <string>
#include <vector>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ggx.cuh"

__global__ void test_uniform_hemisphere(float3* ret, int n){
	if(blockIdx.x*blockDim.x + threadIdx.x > 0) return;

	RNG rng(0);

	for(int i=0; i<n; i++){
		ret[i] = sample_uniform_hemisphere(rng.uniform(), rng.uniform());
	}
}


__global__ void test_normalized(float* comp, float* cosine, uint32_t n_v, uint32_t n_m, float alpha){
	uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=n_v) return;

	float th = i*0.5*M_PI/n_v;
	cosine[i] = cos(th);

	comp[i] = GGX::normalization_constraint(make_float3(sin(th), 0, cos(th)), alpha, n_m);
}

__global__ void to_image(uint32_t* pixels, float* g1_projected, float* cosine, const uint32_t w, const uint32_t h){
	uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i >= w || j >= h) return;

	pixels[j*w + i] = 0xff3a3a3a;
	if( j == (int)(h*(1-cosine[i])) ) pixels[j*w + i] = 0xffbb7722;
	if( j == (int)(h*(1-g1_projected[i])) ) pixels[j*w + i] = 0xff2299bb;
}


int main(){
	const int cores = 32;
	std::vector<cudaStream_t> streams(cores);
	for(auto& stream : streams) cudaStreamCreate(&stream);

	const uint32_t w = 800;
	const uint32_t h = 400;

	float* g1_projected_buf;
	cudaMallocManaged(&g1_projected_buf, cores * w*sizeof(float));

	float* cosine_buf;
	cudaMallocManaged(&cosine_buf, cores * w*sizeof(float));

	uint32_t* pixels_buf;
	cudaMallocManaged(&pixels_buf, cores * w*h*sizeof(uint32_t));

	#pragma omp parallel for schedule(dynamic)
	for(int i=0; i<cores; i++){
		const auto& stream = streams[i];
		float* g1_projected = g1_projected_buf + i*w;
		float* cosine = cosine_buf + i*w;
		uint32_t* pixels = pixels_buf + i*w*h;

		float delta = 0.05;
		for(float alpha = delta*i; alpha<1.0; alpha += delta*cores){
			test_normalized <<<512, w/512 + 1, 0, stream>>> (g1_projected, cosine, w, 10000, alpha);

			const dim3 threads(16, 16);
			const dim3 blocks(w/threads.x + 1, h/threads.y + 1);
			to_image <<<blocks, threads, 0, stream>>> (pixels, g1_projected, cosine, w, h);

			cudaStreamSynchronize(stream);

			std::string out = "result_gpu/projected_area_gpu_" + std::to_string(alpha) + ".png";
			stbi_write_png(out.c_str(), w, h, 4, pixels, w*sizeof(uint32_t));
		}
	}

	cudaFree(g1_projected_buf);
	cudaFree(cosine_buf);
	cudaFree(pixels_buf);
	for(auto& stream : streams)cudaStreamDestroy(stream);
}