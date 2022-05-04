#include <vector>
#include <string>

#include "vector_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ggx.cuh"


int main(){
	const int cores = 32;

	#pragma omp parallel for schedule(dynamic)
	for(int i=0; i<cores; i++){

		const float delta = 0.05;
		for(float alpha=delta*i; alpha<1.0; alpha += delta*cores){

			const int w = 800;
			const int h = 400;
			std::vector<float> cosine(w);
			std::vector<float> g1_projected(w);
			std::vector<uint32_t> pixels(w*h, 0xff3a3a3a);

			for(int i=0; i<w; i++){
				float th = i*0.5*M_PI/w;

				cosine[i] = cos(th);

				const int n_m = 10000;
				g1_projected[i] = GGX::normalization_constraint(make_float3(sin(th), 0, cos(th)), alpha, n_m);


				// write result
				{
					int j = h*(1 - cosine[i]);
					if(j>h-1) j = h-1;
					if(j<0) j = 0;

					// blue
					pixels[j*w + i] = 0xffbb7722;
				}

				{
					int j = h*(1.0 - g1_projected[i]);
					if(j>h-1) j = h-1;
					if(j<0) j = 0;

					// orange
					pixels[j*w + i] = 0xff2299bb;
				}
			}

			std::string out = "result_cpu/projected_area_cpu_" + std::to_string(alpha) + ".png";
			stbi_write_png(out.c_str(), w, h, 4, pixels.data(), w*sizeof(uint32_t));
		}
	}

	return 0;
}