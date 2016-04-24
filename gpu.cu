// Joshua Donnoe, Kyle Evens, and Dominik Haeflinger

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "common.h"

#define NUM_BLOCKS 256
#define NUM_THREADS 256

// from https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
// TODO edit to handle struct/tree
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

  //if(tid < 32) warpReduce(sdata, tid);
  if(tid == 0 ) g_odata[blockIdx.x] = sdata[0];
}
// end from

__global__ void reduce(edge_t* src, edge_t* dest, int ne){
  // Thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= ne) return;

  edge_t* left = &src[tid * 2];
  edge_t* right = left + 1;
  *dest[tid] = left->distance < right->distance ? *left : *right;
}

// Calculates x position in matrix
__device__ void calcXPos(unsigned short *x, int adjIndex, float adjN){
  *x = (unsigned short)(floor(adjN - sqrt(pow(adjN, 2) - adjIndex)));
}

// Calculates y position in matrix
__device__ void calcYPos(unsigned short *y, int adjIndex, float adjN, int x){
  *y = (unsigned short)(adjIndex + (x * (x + adjN)) / 2);
}

// Calculates index in array from position in matrix
__device__ void calcArrayIndex(int *index, int adjN, int adjY, int x){
  *index = (int)((x * (adjN - x) + adjY) / 2);
}

// Calculate the position in the matrix
__device__ void calcPosInMatrix(int index, int n, unsigned short *x, unsigned short *y){
  calcXPos(x, index * 2, n - (.5f));
  calcYPos(y, index + 1, 3 - 2 * n, *x);
}

// Calcuate edges between all points
__global__ void calculateEdge(edge_t* edges, point_t* points, int n){
  // Thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  edge_t *e = &edges[tid];
  calcPosInMatrix(tid, n, &(e->tree1), &(e->tree2));
  point_t *xp = &points[(e->tree1)];
  point_t *yp = &points[(e->tree2)];

  float sum = 0;
  for (int i = 0; i < DIM; i++) {
    float delta = xp->coordinates[i] - yp->coordinates[i];
    sum += delta * delta;
  }
  e->distance = sqrt(sum);
  printf("tid: %d - e->1: %d - e->2: %d - e->d: %f\n\txp->x: %f - xp->y: %f - yp->x: %f - yp->y: %f",
          tid, e->tree1, e->tree2, e->distance, xp->coordinates[0], xp->coordinates[1], yp->coordinates[0], yp->coordinates[1]);
}

// main duh
int main(int argc, char **argv) {

  cudaThreadSynchronize();

  if( find_option( argc, argv, "-h" ) >= 0 )
  {
      printf( "Options:\n" );
      printf( "-h to see this help\n" );
      printf( "-n <int> to set the number of particles\n" );
      printf( "-o <filename> to specify the output file name\n" );
      printf( "-s <filename> to specify the summary output file name\n" );
      return 0;
  }

  int n = read_int(argc, argv, "-n", 1000);

  char *savename = read_string(argc, argv, "-o", NULL);
  char *sumname = read_string(argc, argv, "-s", NULL);

  FILE *fsave = savename ? fopen(savename, "w") : NULL;
  FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

  ///int n = 3;

  // GPU point data tructure
  edge_t * d_edges;
  cudaMalloc((void **) &d_edges, 7 * (n * n - n));
  // GPU point data structure
  point_t * d_points = (point_t *)(((void *) d_edges) + 4 * (n * n - n));
  edge_t* half = (edge_t*)d_points;
  edge_t* quarter = ((void*)half) + 2 * (n * n - n);

  double init_time = read_timer();
  // Initialize points
  curandGenerator_t gen; // Random number generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); // Initialize generator
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); // Set generator's seed
  curandGenerateUniform(gen, (float*)d_points, n * DIM); // Generate n random numbers in d_points

  // Initialize edges
  calculateEdge <<< NUM_BLOCKS, NUM_THREADS >>> (d_edges, d_points, n);

  cudaThreadSynchronize();
  init_time = read_timer() - init_time;
  double reduce_time = read_timer();

  // Reduce tree
  edge_t* smallest = malloc(sizeof(edge_t));
  for (int numEdgesSel = n - 1; numEdgesSel-- > 0;) {
    int numEdgesRed = (n * n - n) / 2;
    reduce <<< NUM_BLOCKS, NUM_THREADS >>> (edges, half, numEdgesRed);
    for(; numEdgesRed >= 4; numEdgesRed / 4){
      reduce <<< NUM_BLOCKS, NUM_THREADS >>> (half, quarter, numEdgesRed / 2);
      reduce <<< NUM_BLOCKS, NUM_THREADS >>> (quarter, half, numEdgesRed / 4);
    }

    if(numEdgesRed == 3){ // 3 elements in half
      reduce <<< 1, 1 >>> (half + 1, half + 1, 2);
    }
    if(numEdgesRed == 2){ // 2 elements in half
      reduce <<< 1, 1 >>> (half, half, 2);
    }
    cudaMemcpy((void*)smallest, (const void*)half, sizeof(edge_t), cudaMemcpyDeviceToHost);
    printf("Smallest %d: %f", numEdgesSel, smallest->distance);
    break;
  }

  cudaThreadSynchronize();
  reduce_time = read_timer() - reduce_time;

  printf("Initialization time = %g seconds\n", init_time);
  printf("n = %d, Reduction time = %g seconds\n", n, reduce_time);

  /*
  if (fsum)
  {
    fprintf(fsum, "%d %lf \n", n, reduce_time);
  }

  if (fsum)
  {
    fclose(fsum);
  }
  */

  cudaFree(d_edges);

  if (fsave)
  {
    fclose(fsave);
  }

  return 0;
}
