/*
  This code for the mixed membership community project was written by Mohammad Umar Hakeem and Niranjan U N and
  are copyrighted under the (lesser) GPL:
  Copyright (C) 2013 Mohammad Umar Hakeem and Niranjan U N
  This program is free software; you can redistribute it and/or modify it under the terms of the
  GNU Lesser General Public License as published by the Free Software Foundation;
  version 3.0 or later. This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE.
  See the GNU Lesser General Public License for more details. You should have received a copy of
  the GNU Lesser General Public License along with this program;
  if not, write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
  02111-1307, USA.
  The authors may be contacted via email at: mhakeem(at)uci(.)edu , un(.)niranjan(at)uci(.)edu
*/

#include <stdio.h>
#include <cuda.h>
#include <curand.h>

// for the curand cuda error handling
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


int gen_rnd_mat(double *hostData, size_t n, double mean, double stddev)
{
    curandGenerator_t gen;
    double *devData;

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)time(NULL)));

    /* Generate n doubles on device */
    CURAND_CALL(curandGenerateNormalDouble(gen, devData, n, mean, stddev));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n*sizeof(double), cudaMemcpyDeviceToHost));

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    return EXIT_SUCCESS;
}
