// Future one-GPU numerical gate. NOT the TP2/model qualification.
// Build with nvcc --fmad=false --ftz=false -O3 -arch=sm_121 kernel_check.cu -o kernel_check
// Execution is explicit: ./kernel_check --execute-on-idle-gpu
#include "runtime_directions.cu"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

static void checked(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result)); std::exit(1);
    }
}
int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--execute-on-idle-gpu")) {
        std::fprintf(stderr, "Not executed: requires --execute-on-idle-gpu after fleet preflight\n"); return 2;
    }
    size_t free_bytes, total_bytes;
    checked(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < (16ULL << 30)) { std::fprintf(stderr, "16 GiB reserve required\n"); return 2; }
    constexpr int width = 5120;
    std::vector<float> r(width);
    double norm = 0;
    for (int j=0; j<width; ++j) { r[j]=std::sin(double(j+1)); norm+=double(r[j])*r[j]; }
    for (float& x:r) x/=std::sqrt(norm);
    float* dr; checked(cudaMalloc(&dr, width*sizeof(float)));
    checked(cudaMemcpy(dr, r.data(), width*sizeof(float), cudaMemcpyHostToDevice));
    double max_error=0; int cases=0;
    for (int rows: {1,17,129,4096}) {
        const size_t count=size_t(rows)*width;
        std::vector<float> y(count), got(count);
        for (size_t i=0; i<count; ++i) y[i]=std::sin(double(i)*0.13)*3.0;
        float* dy; checked(cudaMalloc(&dy,count*sizeof(float)));
        for (float alpha: {0.0f,0.25f,0.5f,1.0f}) {
            checked(cudaMemcpy(dy,y.data(),count*sizeof(float),cudaMemcpyHostToDevice));
            tf_runtime_direction_f32<<<rows,256>>>(dy,dr,alpha,rows);
            checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
            checked(cudaMemcpy(got.data(),dy,count*sizeof(float),cudaMemcpyDeviceToHost));
            if (alpha==0 && std::memcmp(y.data(),got.data(),count*sizeof(float))) return 3;
            for (int row=0; row<rows; ++row) {
                double dot=0;
                for (int j=0; j<width; ++j) dot+=double(y[size_t(row)*width+j])*r[j];
                for (int j=0; j<width; ++j) {
                    const size_t i=size_t(row)*width+j;
                    const double expected=y[i]-double(alpha)*r[j]*dot;
                    const double error=std::abs(got[i]-expected);
                    max_error=std::max(max_error,error);
                    if (!std::isfinite(got[i]) || error>2e-5+3e-5*std::abs(expected)) return 4;
                }
            }
            ++cases;
        }
        checked(cudaFree(dy));
    }
    checked(cudaFree(dr));
    std::printf("{\"numerical_cases\":%d,\"max_abs_error\":%.10g,\"model_qualified\":false}\n",cases,max_error);
    return 0;
}
