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


// c code for post-processing after stpm; uses same header file and compile flags
// entire code needs global memory buffers statically declared to avoid malloc errors and time in dynamic allocation
#include "stpm.h"

// externs from gemm_pinv_nys.c
extern culaStatus culaBlockDgemm(char, char, int, int, int, double, double *, int, double *, int, double, double *, int);

// externs from pre_proc.c
extern double W_buff[NA*NCOM];
extern double Z_B_buff[NA*NB];
extern double G_XA_buff[NA*NX];

// externs from stpm.c
extern double Glob_Buffer_Mat_5[NCOM*NCOM];  // contains final eigenvector matrix calculated in stpm.c
extern double Glob_Buffer_Mat_8[NCOM*NCOM];  // eigenvalue matrix for post processing

double eval_mat_buff[NCOM*NCOM]; // matrix of eigenvalues buffer
double W_evec_eval_buff[NA*NCOM]; // W * eigenvector matrix * diag(inverse eigenvalues)
double Z_B_T_W_evec_eval_buff[NB*NCOM]; // intermediate for Pi_A
double Pi_X_buff[NCOM*NX];

#ifdef SYNTHETIC
double G_BA_buff[NB*NA], G_CA_buff[NC*NA]; // complement matrices
double Pi_A_buff[NCOM*NA], Pi_B_buff[NCOM*NB], Pi_C_buff[NCOM*NC]; // output matrices written into file for hypothesis testing
#endif

// function to write a general matrix / vector or especially, the estimated membership submatrices; note: this writes the matrix in its vectorized form (column-major); can also be used for debugging or saving a snapshot of the algorithm progress
void write_mat(char *file_name, char *mat_name, double *mat_ptr, int N) // output file name, matrix name, pointer to matrix buffer, number of elements of the matrix
{
  int i; // loop counter
  printf("writing %s\n", mat_name); fflush(stdout);
  FILE *file_ptr = fopen(file_name, "w");
  if(file_ptr == NULL) // exception handling if writing mat_name fails
  {
    printf("writing %s failed\n", mat_name);
    culaShutdown();
    exit(1);
  }
  for(i=0; i<N; i++)
    fprintf(file_ptr, "%5.25e\n", *(mat_ptr+i));
  fclose(file_ptr);
}

// function for post-processing; here, exit(1) stands for file i/o error, exit(2) stands for cula error
void post_proc()
{
  double *evec_mat_ptr = Glob_Buffer_Mat_5, *eval_mat_ptr = Glob_Buffer_Mat_8; // matrix of eigenvectors and matrix of eigenvalues pointers
  double *W_ptr = W_buff; // whitening matrix pointer
  double *W_evec_eval_ptr = W_evec_eval_buff; // pointer to product buffer for intermediate buffer post processing
  double *Z_B_T_W_evec_eval_ptr = Z_B_T_W_evec_eval_buff; // intermediate for Pi_A
  double *Pi_X_ptr = Pi_X_buff;
# ifdef SYNTHETIC
  double *G_BA_ptr = G_BA_buff, *G_CA_ptr = G_CA_buff; // complement matrix pointers
  double *Pi_A_ptr = Pi_A_buff, *Pi_B_ptr = Pi_B_buff, *Pi_C_ptr = Pi_C_buff; // recovered Pi matrices
# endif
  double *Z_B_ptr = Z_B_buff;
  double *G_XA_ptr = G_XA_buff;


# ifdef SYNTHETIC
  printf("reading G_BA\n"); fflush(stdout);
  read_G((char *)FILE_BA, "G_BA", G_BA_ptr, (int)NB);

  printf("reading G_CA\n"); fflush(stdout);
  read_G((char *)FILE_BA, "G_CA", G_CA_ptr, (int)NC);
# endif


  printf("computing whitening matrix multiplied by eigenvector matrix\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NCOM, NCOM, 1, W_ptr, NA, evec_mat_ptr, NCOM, 0, W_evec_eval_ptr, NA), "culaBlockDgemm", "W * evecs, i.e., W_evec_eval (step 1)"); // W * evecs

  printf("computing (W * evecs) from previous step multiplied by inverse eigenvalue matrix\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NCOM, NCOM, 1, W_evec_eval_ptr, NA, eval_mat_ptr, NCOM, 0, W_evec_eval_ptr, NA), "culaBlockDgemm", "(W * evecs) * inv_evals, i.e., W_evec_eval"); // (W * evecs) * inv_evals

  printf("computing Pi_X\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 'n', NCOM, NX, NA, 1, W_evec_eval_ptr, NA, G_XA_ptr, NA, 0, Pi_X_ptr, NCOM), "culaBlockDgemm", "Pi_X"); // ((W * evecs) * inv_evals)^T * (G_XA)^T - note: G_XA is already transposed in pre_proc.c; this line computes Pi_X (for real data p_value)

# ifdef SYNTHETIC
  printf("computing Pi_A (step 1)\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 'n', NB, NCOM, NA, 1, Z_B_ptr, NA, W_evec_eval_ptr, NA, 0, Z_B_T_W_evec_eval_ptr, NB), "culaBlockDgemm", "Z_B' * ((W * evecs) * inv_evals)"); // for Pi_A: Z_B' * ((W * evecs) * inv_evals)

  printf("computing Pi_A\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 'n', NCOM, NA, NB, 1, Z_B_T_W_evec_eval_ptr, NB, G_BA_ptr, NB, 0, Pi_A_ptr, NCOM), "culaBlockDgemm", "Pi_A"); // Pi_A finally calculated

  printf("computing Pi_B\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NCOM, NB, NA, 1, W_evec_eval_ptr, NA, G_BA_ptr, NB, 0, Pi_B_ptr, NCOM), "culaBlockDgemm", "Pi_B"); // Pi_B = ((W * evecs) * inv_evals)^T * (G_BA)^T is computed

  printf("computing Pi_C\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NCOM, NC, NA, 1, W_evec_eval_ptr, NA, G_CA_ptr, NC, 0, Pi_C_ptr, NCOM), "culaBlockDgemm", "Pi_C"); // Pi_C = ((W * evecs) * inv_evals)^T * (G_CA)^T is computed
# endif


  write_mat((char *)PI_X, "Pi_X", Pi_X_ptr, (int)(NCOM*NX));
# ifdef SYNTHETIC
  write_mat((char *)PI_A, "Pi_A", Pi_A_ptr, (int)(NCOM*NA));
  write_mat((char *)PI_B, "Pi_B", Pi_B_ptr, (int)(NCOM*NB));
  write_mat((char *)PI_C, "Pi_C", Pi_C_ptr, (int)(NCOM*NC));
# endif


  printf("post-processing completed; ready for hypothesis testing and interpretation\n");
}
