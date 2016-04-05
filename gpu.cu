// Joshua Donnoe, Kyle Evens, and Dominik Haeflinger

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define NUM_THREADS 256

// from https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n){
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSie = blockSize * 2 * gridDim.x;
  sdata[tid] = 0;

  while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSie; }
  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

  if(tid < 32) warpReduce(sdata, tid);
  if(tid == 0 ) g_odata[blockIdxx] = sdata[0];
}
// end from

// generate graph/distance matrix


int main() {


  return 0;
}

/*
  struct edge
    int tree1
    int tree2
    double distance

*/
