----------------------------------------------------
Compatibility requirements:-
----------------------------------------------------
This code was developed and tested using CentOS 6.4, CULA-Dense R16a, CUDA 5.0, GCC v4.4.7, Intel MKL 11.0 (only SVDs were used for testing purposes, but the MKL function calls are not an active part of the code). Other system configurations are given in the appendix of paper "Fast Detection of Overlapping Communities via Online Tensor Methods on GPUs" by Furong Huang, Niranjan U N, Mohammad Umar Hakeem,  Prateek Verma, Animashree Anandkumar.


----------------------------------------------------
Input and output formats:-
----------------------------------------------------
The input file names and path need to be set in stpm.h. The input files are the matrices G_XA, G_XB, G_XC where X, A, B, C are the disjoint partitions of the nodes, i.e., corresponding adjacency submatrices in the COO sparse matrix format (if input matrices are binary, then enable the BINARY flag in stpm.h so that only 2 columns are assumed, else if there are 3 columns as in the weighted case then enable the EXPECTED flag in stpm.h). The indexing starts from 1 (MATLAB style), not 0 (C stlye) - this conversion is done by the file reading function in pre_proc.c. If ground truth is available, then the matrices G_BA and G_CA are also provided to get all the Pi matrices (membership estimates). The output from post_proc.c are Pi_X, Pi_A, Pi_B, Pi_C matrices which can be concatenated to get the overall Pi matrix, i.e., the membership estimates for all the nodes in the network. These are written to file in the column-major order as a single vector, which need to reshaped to a matrix of appropriate size for p-value computation. Other details like running time, etc are written in the sim_info file.


----------------------------------------------------
Some code conventions and descriptions:-
----------------------------------------------------
The entire code is in double precision.

makefile:
This contains all the linker and flag setting to compile the code files below and produce the executable named "algo".

stpm.h:
This is the header file that holds all the libraries and global define settings for the CULA C codes. However, this does not include the libraries necessary for the CUDA codes as these are separately compiled using nvcc and linked later to the CULA codes, which are written in C and compiled using GCC. This file contains all the parameters like Dirichlet parameter, learning rate, numerical tolerance, maximum number of iterations, etc which can be set. Refer to the community paper for the dimensions that are defined and also the variable names.

stpm.c:
This file contains the CULA-device code for the stochastic gradient descent. The global buffers from pre-processing are mapped on Glob_buff_ variables and are pointed to in the device memory in the GPU_Task function. The main function of the C code is in this file and has an optional code in while loop containing 2 CPU threads - one that executed the GPU_Task and the other that listens on the first to serve as a software interrupt. For the exact the manipulation of buffers, refer to the appendix of the community paper.

pre_proc.c and post_proc.c:
These use large global buffers statically declared to avoid malloc failures. The buffer names are suffixed with _buff and the correspoding pointers which are passed to BLAS/LAPACK functions are suffixed with _ptr. pre_proc.c takes care of the matrix operations till the whitening matrix is calculated. post_proc.c computes the membership estimates from the eigenvalues and eigenvectors obtained from stpm.c.

gemm_pinv_nys.c:
This contains the support functions written for handling DBLP data, which is too large to fit in the GPU memory. This file includes
1. a partitioned implementation of the BLAS function DGEMM using the culaDgemm as the building block. The BLOCK_SIZE is set in stpm.h.
2. Ben-Israel and Cohen iterative pseudoinverse computation routine (which has not been used in the community paper)
3. Randomized Nystrom method for SVD (needed for pseudoinverse and whitening).

wrappers.cu:
This file implements CUDA kernels (support functions for stpm.c are suffixed with _CudaKer and the function names are self-explanatory) for supporting functionalities such as taking the elementwise square root of a vector, etc.

curand_rand_mat.cu:
This implements is a CUDA code to generate iid samples from N(0, 1) to fill up a random matrix for the Nystrom method.

