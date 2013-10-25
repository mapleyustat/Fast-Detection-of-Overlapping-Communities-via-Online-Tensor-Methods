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


// header file containing all the includes / defines needed for the stochastic tensor power method
// header guard
#ifndef STPM_H
#define STPM_H

// header files and required libraries / APIs
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cula.h>
#include <cula_blas.h>
#include <cula_lapack.h>
#include <cula_device.h>
#include <cula_lapack_device.h>
#include <cula_blas_device.h>
#include <sys/time.h>
#include <math.h>
//#include <mkl.h> // uncomment if using MKL SVD
#include <pthread.h>
#include <cuda_runtime.h>
// to get standard normal numbers for nystrom method
#include <cuda.h>
#include <curand.h>

// max and min macros (because -lm does not have these in the system)
#define max(x, y) (x>y?x:y)
#define min(x, y) (x<y?x:y)

// path of the datasets on the disk
#define FOLDER_NAME "DBLP_116317/" // optionally define at compile time with -DFOLDER_NAME=f($loop_var)
#define TXT ".txt"

// numeric to string pre-processor converter
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// flags and buffer sizes
#define SYNTHETIC // use when ground truth is known; optionally define at compile time with -DSYNTHETIC=$constant
#define BINARY // use when there are 2 columns in the adjacency submatrices; optionally define at compile time with -DBINARY=$constant
//#define EXPECTED // use when there are 3 columns in the sparse adjacency submatrices, eg, expected case / yelp data; optionally define at compile time with -DEXPECTED=$constant
#define STR_BUF_SIZE 1000
#define ITERS_BEFORE_CONV_TEST 1
#define FROB_NORM_VALS_BUFF_SIZE 100000 // arbitrarily large buffer to store the frobenius norm double values

// input files
#define FILE_A FOLDER_NAME "DBLP_Gx_a" TXT
#define FILE_B FOLDER_NAME "DBLP_Gx_b" TXT
#define FILE_C FOLDER_NAME "DBLP_Gx_c" TXT
#ifdef SYNTHETIC
# define FILE_BA FOLDER_NAME "DBLP_Gb_a" TXT
# define FILE_CA FOLDER_NAME "DBLP_Gc_a" TXT
#endif

// code parameters
#define NX 29079 // number of nodes in partition X; optionally define at compile time with -DNX=f($loop_var)
#define NA 29080 // number of nodes in partition A; optionally define at compile time with -DNA=f($loop_var)
#define NB 29079 // number of nodes in partition B; optionally define at compile time with -DNB=f($loop_var)
#define NC 29079 // number of nodes in partition C; optionally define at compile time with -DNC=f($loop_var)
#define NCOM 100 // number of communities (our estimate not the true one); optionally define at compile time with -DNCOM=f($loop_var)
#define NMAX max(max(max(NX, NA), NB), NC)
#define ALPHA_0 1.0 // dirichlet parameter; optionally define at compile time with -DALPHA_0=f($loop_var)
#define LEARN_RATE 1e-3 // learning rate; optionally define at compile time with -DLEARN_RATE=f($loop_var)

// output files
#define PI_X FOLDER_NAME "Pi_X_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
#ifdef SYNTHETIC
# define PI_A FOLDER_NAME "Pi_A_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
# define PI_B FOLDER_NAME "Pi_B_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
# define PI_C FOLDER_NAME "Pi_C_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
#endif
#define EVECS FOLDER_NAME "evecs_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
#define EVALS FOLDER_NAME "evals_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
#define SIM_INFO FOLDER_NAME "sim_info_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT
#define FROB_NORM FOLDER_NAME "frob_norm_" STR(NCOM) "_" STR(ALPHA_0) "_" STR(LEARN_RATE) TXT

// numerical stability and convergence check issues
#define THRESH 1//1e-8 // threshold for stpm convergence
#define PINV_TOL 1e-6 // numerical tolerance for taking inverse / reciprocal
#define SGD_ITER_MIN NX // minimum number of stpm iterations
#define SGD_ITER_MAX NX+2 // maximum number of stpm iterations

// support for dblp
// the block size for partitioned (d)gemm
#define BLOCK_SIZE 10000
// numerical parameters for iterative pseudoinverse
#define MAX_PINV_ITER 45
#define MIN_PINV_ITER 25
#define PINV_CVG_TST 1e-3
#define PINV_INIT_COEFF 1e-8

#endif
