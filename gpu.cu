// Joshua Donnoe, Kyle Evans, and Dominik Haeflinger

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "common.h"

#define NUM_THREADS 256

//k is the number of edges originally
//half is the new number of edges. which is half of e rounded up
__global__ void reduce(edge_t* src, edge_t* dest, int e, int half){
  // Thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= half) return;

  edge_t* left = &src[tid * 2];
  edge_t* right = left + 1;

  if (right == src + e) {
    memcpy((void*) &dest[tid], (const void*) left, 8);
  }
  else {
    memcpy((void*) &dest[tid], (const void*) ((left->distance < right->distance)) ? left : right, 8);
  }
}

// Calculates x position in matrix
__device__ void calcXPos(unsigned short *x, int adjIndex, float adjN, float adjN2){
  *x = (unsigned short)(floor(adjN - sqrt(adjN2 - adjIndex)));
}

// Calculates y position in matrix
__device__ void calcYPos(unsigned short *y, int adjIndex, float adjN, int x){
  *y = (unsigned short)(adjIndex + (x * (x + adjN)) / 2);
}

// Calculate the position in the matrix
__device__ void calcPosInMatrix(int index, unsigned short *x, unsigned short *y, float adjNX, float adjNX2, int adjNY){
  calcXPos(x, 2 * index, adjNX, adjNX2);
  calcYPos(y, index + 1, adjNY, *x);
}

// Calcuate edges between all points
__global__ void calculateEdge(edge_t* edges, point_t* points, int e, float adjNX, float adjNX2, int adjNY){
  // Thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= e) return;

  edge_t *edge = &edges[tid];
  calcPosInMatrix(tid, &(edge->tree1), &(edge->tree2), adjNX, adjNX2, adjNY);
  point_t *xp = &points[(edge->tree1)]; // point at x value
  point_t *yp = &points[(edge->tree2)]; // point at y value

  float sum = 0;
  for (int i = 0; i < DIM; i++) {
    float delta = xp->coordinates[i] - yp->coordinates[i];
    sum += delta * delta;
  }
  edge->distance = sqrt(sum);
}

__global__ void updateTree(edge_t* edges, int e, unsigned short o, unsigned short n) {
  // Thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= 2 * e) return;

  edge_t* edge = &edges[tid%e];
  unsigned short *tree = (tid > e) ? &edge->tree1 : &edge->tree2;
  if (*tree == o) {
    *tree = n;
  }
}

__global__ void updateTree2(edge_t* edges, int e, unsigned short o, unsigned short n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= e) return;

  edge_t* edge = &edges[tid];
  if (edge->tree2 == o) {
    edge->tree2 = n;
  }
}

__global__ void updateDistance(edge_t* edges, int e) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= e) return;

  edge_t* edge = &edges[tid];
  if (edge->tree1 == edge->tree2) {
    edge->distance = 1/0.;
  }
}

// main duh
int main(int argc, char **argv) {

  const int N = read_int(argc, argv, "-N", 1000);
  const int E = N * (N-1)/2;
  const int NUM_BLOCKS = ceil(E / 256);

  //pointers
  edge_t* d_edges;
  cudaMalloc((void **) &d_edges, 7 * (N * N - N)+ 16);
  point_t * d_points = (point_t *)(((void *) d_edges) + 4 * (N * N - N));
  edge_t* half = (edge_t*)d_points;
  edge_t* quarter = (edge_t*)(((void*)half) + 2 * (N * N - N) + 8);
  edge_t smallest;

  //curand
  curandGenerator_t gen; // Random number generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); // Initialize generator
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL)); // Set generator's seed

  //adjusted values
  float adjNX = N - .5f;
  float adjNX2 = adjNX * adjNX;
  int adjNY = 3 - 2*N;

  float sum = 0;
  // Perform calculations 1000 times
  for (int i = 0; i < 1000 ; i++) {
    curandGenerateUniform(gen, (float*)d_points, N * DIM); // Generate n random numbers in d_points
    calculateEdge <<< NUM_BLOCKS, NUM_THREADS >>> (d_edges, d_points, E, adjNX, adjNX2, adjNY);

    for (int numEdgesSel = N - 1; numEdgesSel-- > 0;) {
      cudaThreadSynchronize();

      int numEdgesRed = E;
      reduce <<< NUM_BLOCKS, NUM_THREADS >>> (d_edges, half, numEdgesRed, (numEdgesRed + 1) / 2);
      numEdgesRed = (numEdgesRed + 1) / 2;

      while(numEdgesRed > 1){
        cudaThreadSynchronize();

        reduce <<< NUM_BLOCKS, NUM_THREADS >>> (half, quarter, numEdgesRed, (numEdgesRed + 1) / 2);
        numEdgesRed = (numEdgesRed + 1) / 2;

        cudaThreadSynchronize();

        reduce <<< NUM_BLOCKS, NUM_THREADS >>> (quarter, half, numEdgesRed, (numEdgesRed + 1) / 2);
        numEdgesRed = (numEdgesRed + 1) / 2;
      }

      cudaMemcpy((void*)&smallest, (const void*)half, sizeof(edge_t), cudaMemcpyDeviceToHost);
      sum += smallest.distance;

      updateTree <<< NUM_BLOCKS, NUM_THREADS >>> (d_edges, E, smallest.tree1, smallest.tree2);
      updateDistance <<< NUM_BLOCKS, NUM_THREADS >>> (d_edges, E);
    }
  }

  printf("sum %f\n", sum/1000);
  // Clean-up
  cudaFree(d_edges);
  curandDestroyGenerator(gen);
  return 0;
}
