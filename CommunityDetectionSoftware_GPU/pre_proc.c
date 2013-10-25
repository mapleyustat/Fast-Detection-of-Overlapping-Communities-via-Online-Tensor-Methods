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


// c code for pre-processing before stpm; uses same header file and compile flags
// entire code needs global memory buffers statically declared to avoid malloc errors and time in dynamic allocation
#include "stpm.h"

// extern dblp support functions from gemm_pinv_nys.c
extern culaStatus culaBlockDgemm(char, char, int, int, int, double, double *, int, double *, int, double, double *, int);
extern culaStatus pinv(int, int, double *, double *);
extern culaStatus pinv_nys_asym(int, int, int, double *, double *);
extern culaStatus nystrom_whitening(int, int, double *, double *);

// note: we need NA \times NX... matrices; also matlab to c indexing conversion done
double G_XA_buff[NA*NX], G_XB_buff[NB*NX], G_XC_buff[NC*NX]; // sparse adjacency matrices stored as NA \times NX etc for efficient sampling for stpm
double G_XB_til_buff[NX*NA], G_XC_til_buff[NX*NA]; // tilde matrices
double G_XA_white_buff[NCOM*NX], G_XB_til_white_buff[NCOM*NX], G_XC_til_white_buff[NCOM*NX]; // whitened matrices
double M2_al0_buff[NA*NA]; // moment 2 as in theorem 3.6, tensor paper
double G_XA_al0_buff[NA*NX], G_XB_al0_buff[NB*NX], G_XC_al0_buff[NC*NX]; // G^alpha0
double mu_A_buff[NA], mu_B_buff[NB], mu_C_buff[NC]; // mean vectors
double mu_B_til_buff[NA], mu_C_til_buff[NA]; // mean tilde vectors
double mu_A_white_buff[NCOM], mu_B_til_white_buff[NCOM], mu_C_til_white_buff[NCOM]; // buffers for whitened mean vectors
double mu_A_mu_A_T_buff[NA*NA]; // square matrix of outer product
double Z_B_buff[NA*NB], Z_B_num_buff[NA*NC], Z_B_den_buff[NB*NC], Z_C_buff[NA*NC], Z_C_num_buff[NA*NB];//, Z_C_den_buff[NC*NB]; // commented Z_C_den because same as transpose(Z_B_den)
double ones_buff[NX], eye_buff[NA*NA]; // vector of all ones for averaging; identity buffer
//double pinv_buff[NMAX*NMAX];//, l_svec_mat_buff[NMAX*NMAX], r_svec_mat_T_buff[NMAX*NMAX], sval_vec_buff[NMAX], sval_mat_buff[NMAX*NMAX]; // uncomment and use this line if using culadgesvd, as long as it obeys the cula-quadro version of hooke's law
double pinv_buff[NC*NB];//, l_svec_mat_buff[NMAX*NMAX], r_svec_mat_T_buff[NMAX*NMAX], sval_vec_buff[NMAX], sval_mat_buff[NMAX*NMAX]; // uncoment if using culadgesvd within its breaking point
double W_buff[NA*NCOM]; // buffer for whitening matrix
//double sup_buff[NMAX]; // superb parameter for mkl svd; if not using mkl, not needed
struct timeval start_timeval_svd1, stop_timeval_svd1;
struct timeval start_timeval_svd2, stop_timeval_svd2;


// function to read the adjacency submatrices from file (when stored in two-column or three-column sparse matrix format with doubleing point entries); note: this reads, for example, G_XA^T instead of G_XA
void read_G(char *file_name, char *G_name, double *G_ptr, int N) // input file name, matrix name, pointer to matrix buffer, (NA or NB or NC)
{
  double r_idx, c_idx; // row and column indices - matlab style
  printf("reading %s\n", G_name); fflush(stdout);
  FILE *file_ptr = fopen(file_name, "r"); // opening G_name
  if(file_ptr == NULL) // exception handling if reading G_name fails
  {
    printf("reading %s adjacency submatrix failed\n", G_name);
    culaShutdown();
    exit(1);
  }
  while(!feof(file_ptr)) // reading G_name
  {
    fscanf(file_ptr, "%lf", &c_idx); // note: since we need (NA or NB or NC) \times NX, we read the column index first, then usual column-major
    fscanf(file_ptr, "%lf", &r_idx); // now, NROWS = (NA or NB or NC) (y-axis), NCOLS = NX (x-axis)
# ifdef EXPECTED
    fscanf(file_ptr, "%lf", G_ptr+(int)((c_idx-1)*N+(r_idx-1)));
# endif
# ifdef BINARY
    *(G_ptr+(int)((c_idx-1)*N+(r_idx-1))) = 1;
# endif
    /*
    // optional printing to check file read is faithful (used for debugging)
    printf("%lf\n", c_idx);
    printf("%lf\n", r_idx);
    printf("%lf\n", *(G_ptr+(int)((c_idx-1)*N+(r_idx-1))));
    culaShutdown();
    exit(0);
    */
  }
  fclose(file_ptr);
}

// function to print vectors and matrices for debugging
void print_mat(int I, int J, double *mat_ptr) // number of rows, columns, pointer to the column-major buffer
{
  int i, j;
  for(i=0; i<I; i++)
  {
    for(j=0; j<J; j++)
      printf("%lf ", *(mat_ptr+(j*I+i)));
//      printf("%2.5e ", *(mat_ptr+(j*I+i))); // use if decimal display precision needs to be specified
    printf("\n");
  }
}

// function for cula exception handling - make sure that the char * arguments are not NULL while calling
void cula_exception(culaStatus cula_err, char *cula_func, char *term) // error status, cula function that fails, term in the algorithm
{
  int cula_info; // identifier for the cula error
  char cula_msg[256]; // buffer for storing the cula excepetion message
  if(cula_err != culaNoError)
  {
    cula_info = culaGetErrorInfo();
    culaGetErrorInfoString(cula_err, cula_info, cula_msg, sizeof(cula_msg));
    printf("(cula error) user message: %s for %s failed; cula message: %s\n", cula_func, term, cula_msg);
    fflush(stdout);
    culaShutdown();
    exit(2);
  }
}

// function for pre-processing; here, exit(1) stands for file i/o error, exit(2) stands for cula error
int pre_proc()
{
  cula_exception(culaSelectDevice(1), "culaSelectDevice", "pre_proc gpu selection"); // select which gpu to execute on; 0 is master or primary, 1 is slave or secondary
  printf("calling culaInitialize\n"); fflush(stdout);
  cula_exception(culaInitialize(), "culaInitialize", "pre_proc.c");

  // pointers to global buffers
  double *G_XA_ptr = G_XA_buff, *G_XB_ptr = G_XB_buff, *G_XC_ptr = G_XC_buff; // adjacency matrix pointers
  double *G_XB_til_ptr = G_XB_til_buff, *G_XC_til_ptr = G_XC_til_buff; // tilde pointers
  double *G_XA_white_ptr = G_XA_white_buff, *G_XB_til_white_ptr = G_XB_til_white_buff, *G_XC_til_white_ptr = G_XC_til_white_buff; // whitened matrix related pointers
  double *M2_al0_ptr = M2_al0_buff; // moment 2 pointer
  double *G_XA_al0_ptr = G_XA_al0_buff, *G_XB_al0_ptr = G_XB_al0_buff, *G_XC_al0_ptr = G_XC_al0_buff; // alpha0 matrix pointers
  double *mu_A_ptr = mu_A_buff, *mu_B_ptr = mu_B_buff, *mu_C_ptr = mu_C_buff; // mean vector pointers
  double *mu_B_til_ptr = mu_B_til_buff, *mu_C_til_ptr = mu_C_til_buff; // mean tilde vector pointers
  double *mu_A_white_ptr = mu_A_white_buff, *mu_B_til_white_ptr = mu_B_til_white_buff, *mu_C_til_white_ptr = mu_C_til_white_buff; // pointers to buffers for whitened mean vectors
  double *mu_A_mu_A_T_ptr = mu_A_mu_A_T_buff; // outer product of mean vector A with itself
  double *Z_B_ptr = Z_B_buff, *Z_B_num_ptr = Z_B_num_buff, *Z_B_den_ptr = Z_B_den_buff, *Z_C_ptr = Z_C_buff, *Z_C_num_ptr = Z_C_num_buff;//, *Z_C_den_ptr = Z_C_den_buff;
  double *ones_ptr = ones_buff, *eye_ptr = eye_buff; // pointer to vector of ones for computing mean of size NX; identity matrix pointer
  double *pinv_ptr = pinv_buff;//, *l_svec_mat_ptr = l_svec_mat_buff, *r_svec_mat_T_ptr = r_svec_mat_T_buff, *sval_vec_ptr = sval_vec_buff, *sval_mat_ptr = sval_mat_buff; // pointers to buffers for pseudoinverse computation
  double *W_ptr = W_buff; // pointers to buffer for whitening matrix using second method (pairs)
  int i, j; // for looping / indexing
  //double *superb = sup_buff; // pointer to superb for mkl svd
  //mkl_set_num_threads(16); // number of cpu threads


  printf("initializing all ones vector of length NX\n"); fflush(stdout);
  for(i=0; i<NX; i++) // using this loop to initialize because memset 1 for double does not hold
    *(ones_ptr+i) = 1;
  printf("initializing identity matrix of size NA x NA\n"); fflush(stdout);
  for(i=0; i<NA*NA; i+=NA+1) // identity for cula products
    *(eye_ptr+i) = 1;


  // read the adjacency submatrices from the dataset
  read_G((char *)FILE_A, "G_XA", G_XA_ptr, (int)NA);
  read_G((char *)FILE_B, "G_XB", G_XB_ptr, (int)NB);
  read_G((char *)FILE_C, "G_XC", G_XC_ptr, (int)NC);


  // compute mean vectors
  printf("computing mu_A\n"); fflush(stdout);
/////  cula_exception(culaDgemv('n', NA, NX, 1/(double)(NX), G_XA_ptr, NA, ones_ptr, 1, 0, mu_A_ptr, 1), "culaDgemv", "mu_A"); // compute mean vector A
  cula_exception(culaBlockDgemm('n', 'n', NA, 1, NX, 1/(double)(NX), G_XA_ptr, NA, ones_ptr, NX, 0, mu_A_ptr, NA), "culaBlockDgemm", "mu_A"); // compute mean vector A using partitioned matrix multiplication

  printf("computing mu_B\n"); fflush(stdout);
/////  cula_exception(culaDgemv('n', NB, NX, 1/(double)(NX), G_XB_ptr, NB, ones_ptr, 1, 0, mu_B_ptr, 1), "culaDgemv", "mu_B"); // compute mean vector B
  cula_exception(culaBlockDgemm('n', 'n', NB, 1, NX, 1/(double)(NX), G_XB_ptr, NB, ones_ptr, NX, 0, mu_B_ptr, NB), "culaBlockDgemm", "mu_B"); // compute mean vector B using partitioned matrix multiplication

  printf("computing mu_C\n"); fflush(stdout);
/////  cula_exception(culaDgemv('n', NC, NX, 1/(double)(NX), G_XC_ptr, NC, ones_ptr, 1, 0, mu_C_ptr, 1), "culaDgemv", "mu_C"); // compute mean vector C
  cula_exception(culaBlockDgemm('n', 'n', NC, 1, NX, 1/(double)(NX), G_XC_ptr, NC, ones_ptr, NX, 0, mu_C_ptr, NC), "culaBlockDgemm", "mu_C"); // compute mean vector C using partitioned matrix multiplication


  // compute Z_B
  printf("computing Z_B_num\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 't', NA, NC, NX, 1/(double)(NX), G_XA_ptr, NA, G_XC_ptr, NC, 0, Z_B_num_ptr, NA), "culaBlockDgemm", "Z_B_num");

  printf("computing Z_B_den\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 't', NB, NC, NX, 1/(double)(NX), G_XB_ptr, NB, G_XC_ptr, NC, 0, Z_B_den_ptr, NB), "culaBlockDgemm", "Z_B_den");


  // note: if the estimated Pi's are 0's, then print and check Z_B_num and den here; if G_XA and G_XC are sparse, then product is even sparser; so, input dataset must be reclustered to have sufficient density; also check sizes to be handled by cula
/****** // this commented pseudoinverse using svd was used for synthetic, facebook and yelp datasets
  printf("computing SVD for Z_B_den\n"); fflush(stdout);
  cula_exception(culaDgesvd('A', 'A', NB, NC, Z_B_den_ptr, NB, sval_vec_ptr, l_svec_mat_ptr, NB, r_svec_mat_T_ptr, NC), "culaDgesvd", "Z_B_den");

///  if mkl svd is needed, using the following 3 lines instead of its cula counterpart
///  gettimeofday(&start_timeval_svd1, NULL);  // Measuring start time for svd1
///  LAPACKE_dgesvd(CblasColMajor, 'A', 'A', NB, NC, Z_B_den_ptr, NB, sval_vec_ptr, l_svec_mat_ptr, NB, r_svec_mat_T_ptr, NC, superb);
///  gettimeofday(&stop_timeval_svd1, NULL);  // Measuring stop time for svd1

  printf("for loop to copy singular values from SVD of Z_B_den\n"); fflush(stdout);
  for(i=0; i<NB; i++) // aliter: use cuda kernel from wrappers.cu but take care of the if check
  {
    if(i>NC)
      break;
    else if(fabs(*(sval_vec_ptr+i)) > PINV_TOL)
      *(sval_mat_ptr+(NB*i+i)) = 1/(*(sval_vec_ptr+i));
  }

  printf("computing pseudoinverse for Z_B_den (step 1)\n"); fflush(stdout);
  cula_exception(culaDgemm('n', 'n', NB, NC, NB, 1, l_svec_mat_ptr, NB, sval_mat_ptr, NB, 0, pinv_ptr, NB), "culaBlockDgemm", "Z_B_den^+ (1), i.e., U*S_-1"); // U*S_-1

  printf("computing pseudoinverse for Z_B_den (step 2)\n"); fflush(stdout);
  cula_exception(culaDgemm('n', 'n', NB, NC, NC, 1, pinv_ptr, NB, r_svec_mat_T_ptr, NC, 0, pinv_ptr, NB), "culaBlockDgemm", "Z_B_den^+ (2), i.e., (U*S_-1)*VT"); // (U*S_-1)*VT - note: this is not the pseudoinverse yet; this is transposed later on directly in the subsequent gemm to get V * S_-1 * U^T
******/

/* // the following 2 lines use the iterative method; very slow even with partitioned gemm
  printf("computing pseudoinverse of Z_B_den using ben-israel iterations\n");
  cula_exception(pinv(NB, NC, Z_B_den_ptr, pinv_ptr), "ben-israel pinv (c malloc must have failed)", "pinv(Z_B_den)");
*/


/****** // this was the code that used pinv_ptr before transposing; this was used for facebook, yelp before partitioning, nystrom, etc.; note: look at the comment block associated with the memset statements below
    cula_exception(culaDgemm('n', 't', NA, NB, NC, 1, Z_B_num_ptr, NA, pinv_ptr, NB, 0, Z_B_ptr, NA), "culaDgemm", "Z_B"); // note: here 2nd arg is 't' because pinv is not yet transposed
******/

/****** // resetting buffers to 0 for reusing below; note: if uncommented, take care that this comes after the pinv_ptr values are used, i.e., insert in the appropriate location
  memset(pinv_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(pinv_ptr, 0, NB*NC*sizeof(double));
******/

  printf("computing pseudoinverse of Z_B_den using nystrom\n");
  cula_exception(pinv_nys_asym(NB, NC, NCOM, Z_B_den_ptr, pinv_ptr), "nystrom pinv (c malloc could have failed)", "pinv(Z_B_den)");

  printf("computing Z_B\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NB, NC, 1, Z_B_num_ptr, NA, pinv_ptr, NC, 0, Z_B_ptr, NA), "culaBlockDgemm", "Z_B"); // note: here 2nd arg was 't' in the previous version of the code because pinv was not yet transposed; this was changed for the nystrom because 2nd arg is n not t as it is already transposed when it is returned from pin_nys_asym function

/******
  // reset buffers
  printf("resetting buffers used for pseudoinverse of Z_B_den\n"); fflush(stdout);
  memset(sval_vec_ptr, 0, NMAX*sizeof(double));
  memset(sval_mat_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(l_svec_mat_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(r_svec_mat_T_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(pinv_ptr, 0, NMAX*NMAX*sizeof(double));
******/

  // compute Z_C
  printf("computing Z_C_num\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 't', NA, NB, NX, 1/(double)(NX), G_XA_ptr, NA, G_XB_ptr, NB, 0, Z_C_num_ptr, NA), "culaBlockDgemm", "Z_C_num");

/* // not needed: same as transpose of Z_B_den; can be used as debugging check
  printf("computing Z_C_den\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 't', NC, NB, NX, 1/(double)(NX), G_XC_ptr, NC, G_XB_ptr, NB, 0, Z_C_den_ptr, NC), "culaBlockDgemm", "Z_C_den");
*/

/****** // this commented pseudoinverse using svd was used for synthetic, facebook and yelp datasets
  printf("computing svd for Z_C_den\n"); fflush(stdout);
  cula_exception(culaDgesvd('A', 'A', NC, NB, Z_C_den_ptr, NC, sval_vec_ptr, l_svec_mat_ptr, NC, r_svec_mat_T_ptr, NB), "culaDgesvd", "Z_C_den");

///  if mkl svd is needed, using the following 3 lines instead of its cula counterpart
///  gettimeofday(&start_timeval_svd2, NULL);  // Measuring start time for svd2
///  LAPACKE_dgesvd(CblasColMajor, 'A', 'A', NC, NB, Z_C_den_ptr, NC, sval_vec_ptr, l_svec_mat_ptr, NC, r_svec_mat_T_ptr, NB, superb);
///  gettimeofday(&stop_timeval_svd2, NULL);  // Measuring stop time for svd2

  printf("for loop to copy singular values from svd of Z_C_den\n"); fflush(stdout);
  for(i=0; i<NC; i++) // aliter: could use cuda kernel from wrappers.cu but take care of the if check
  {
    if(i>NB)
      break;
    else if(fabs(*(sval_vec_ptr+i)) > PINV_TOL)
      *(sval_mat_ptr+(NC*i+i)) = 1/(*(sval_vec_ptr+i));
  }

  printf("computing pseudoinverse for Z_C_den (step 1)\n"); fflush(stdout);
  cula_exception(culaDgemm('n', 'n', NC, NB, NC, 1, l_svec_mat_ptr, NC, sval_mat_ptr, NC, 0, pinv_ptr, NC), "culaBlockDgemm", "Z_C_den^+ (1), i.e., U*S_-1"); // U*S_-1

  printf("computing pseudoinverse for Z_C_den (step 2)\n"); fflush(stdout);
  cula_exception(culaDgemm('n', 'n', NC, NB, NB, 1, pinv_ptr, NC, r_svec_mat_T_ptr, NB, 0, pinv_ptr, NC), "culaBlockDgemm", "Z_C_den^+ (2), i.e., (U*S_-1)*VT"); // (U*S_-1)*VT - note: this is not the pseudoinverse yet; this is transposed later on directly in the subsequent gemm to get V * S_-1 * U^T
******/

/* // ben-israel iterations
  pinv(NB, NC, Z_C_den_ptr, pinv_ptr);
*/

  printf("computing Z_C\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 't', NA, NC, NB, 1, Z_C_num_ptr, NA, pinv_ptr, NC, 0, Z_C_ptr, NA), "culaBlockDgemm", "Z_C"); // note: here 2nd arg is 't' because pinv is not yet transposed; this was changed for the nystrom because 2nd arg is t since we have to transpose pinv(Z_B_den)

/******
  // reset buffers
  printf("resetting buffers used for pseudoinverse of Z_C_den\n"); fflush(stdout);
  memset(sval_vec_ptr, 0, NMAX*sizeof(double));
  memset(sval_mat_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(l_svec_mat_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(r_svec_mat_T_ptr, 0, NMAX*NMAX*sizeof(double));
  memset(pinv_ptr, 0, NMAX*NMAX*sizeof(double));
******/


  // compute tilde matrices
  printf("computing G_XB_til\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NX, NA, NB, 1, G_XB_ptr, NB, Z_B_ptr, NA, 0, G_XB_til_ptr, NX), "culaBlockDgemm", "G_XB_til"); // 1st arg is t as G_XB is NB x NX; 2nd arg is Z_B which is NA \times NB

  printf("computing G_XC_til\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NX, NA, NC, 1, G_XC_ptr, NC, Z_C_ptr, NA, 0, G_XC_til_ptr, NX), "culaBlockDgemm", "G_XC_til"); // 1st arg is t as G_XC is NC x NX; 2nd arg is Z_C which is NA \times NC


  // compute M2_al0
  printf("computing M2_al0 (step 1)\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 'n', NA, NA, NX, 1/(double)(NX), G_XC_til_ptr, NX, G_XB_til_ptr, NX, 0, M2_al0_ptr, NA), "culaBlockDgemm", "M2_al0 (step 1)"); // intermediate M2

  printf("computing mu_A \\otimes mu_A\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NA, 1, 1, mu_A_ptr, NA, mu_A_ptr, 1, 0, mu_A_mu_A_T_ptr, NA), "culaBlockDgemm", "mu_A \\otimes mu_A"); // outer product for shifted term

  printf("for loop for setting the diagonal of mu_A \\otimes mu_A to zero\n"); fflush(stdout);
  for(i=0; i<NA; i++)
    *(mu_A_mu_A_T_ptr+(NA*i+i)) = 0; // setting the diagonal to 0 for centering

  printf("computing M2_al0 (step 2/final step)\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NA, NA, -ALPHA_0/(ALPHA_0+1), eye_ptr, NA, mu_A_mu_A_T_ptr, NA, 1, M2_al0_ptr, NA), "culaBlockDgemm", "M2_al0"); // M2_al0 (NA x NA) computed


  printf("computing whitening matrix using the nystrom method\n"); fflush(stdout);
  cula_exception(nystrom_whitening(NA, NCOM, M2_al0_ptr, W_ptr), "nystrom whitening (c malloc could have failed)", "W");


/******
  cula_exception(culaDgesvd('A', 'A', NA, NA, M2_al0_ptr, NA, sval_vec_ptr, l_svec_mat_ptr, NA, r_svec_mat_T_ptr, NA), "culaDgesvd", "M2_al0"); // k-svd: remember when comparing to matlab's output that if x is singular vector, so is -x; also here, we are actually doing full svd and selecting NCOM singular values

  printf("for loop to copy singular values from svd of M2_al0\n"); fflush(stdout);
  for(i=0; i<NA; i++)
  {
    if(i>NA)
      break;
    else if(fabs(*(sval_vec_ptr+i)) > PINV_TOL)
      *(sval_mat_ptr+(NA*i+i)) = 1/sqrt(*(sval_vec_ptr+i));
  }

  // computing the whitening matrix W
  printf("computing whitening matrix W\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('n', 'n', NA, NCOM, NA, 1, l_svec_mat_ptr, NA, sval_mat_ptr, NA, 0, W_ptr, NA), "culaBlockDgemm", "W"); // whitening matrix W computed
******/


/*
  // uncomment these lines to debug for small test cases to check if the whitening matrix is non-zero / computed correctly
  printf("whitening matrix debug check\n");
  fflush(stdout);
  print_mat(1, NA, W_ptr);
  culaShutdown();
  exit(2);
*/


  // generate the whitened data for stochastic method coded in another c source file - the adjacency matrix buffers are externed in the other source file
  printf("computing whitened adjacency matrix for A\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 'n', NCOM, NX, NA, 1, W_ptr, NA, G_XA_ptr, NA, 0, G_XA_white_ptr, NCOM), "culaBlockDgemm", "G_XA_white"); // G_XA_white

  printf("computing whitened adjacency matrix for B\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NCOM, NX, NA, 1, W_ptr, NA, G_XB_til_ptr, NX, 0, G_XB_til_white_ptr, NCOM), "culaBlockDgemm", "G_XB_til_white"); // G_XB_til_white

  printf("computing whitened adjacency matrix for C\n"); fflush(stdout);
  cula_exception(culaBlockDgemm('t', 't', NCOM, NX, NA, 1, W_ptr, NA, G_XC_til_ptr, NX, 0, G_XC_til_white_ptr, NCOM), "culaBlockDgemm", "G_XC_til_white"); // G_XC_til_white


  printf("computing whitened mean vector for A\n"); fflush(stdout);
/////  cula_exception(culaDgemv('t', NA, NCOM, 1, W_ptr, NA, mu_A_ptr, 1, 0, mu_A_white_ptr, 1), "culaDgemv", "mu_A_white"); // compute whitened mean vector A
  cula_exception(culaBlockDgemm('t', 'n', NCOM, 1, NA, 1, W_ptr, NA, mu_A_ptr, NA, 0, mu_A_white_ptr, NCOM), "culaBlockDgemm", "mu_A_white"); // compute whitened mean vector A using partitioned matrix multiplication

  printf("computing whitened mean vector for B (step 1)\n"); fflush(stdout);
/////  cula_exception(culaDgemv('n', NA, NB, 1, Z_B_ptr, NA, mu_B_ptr, 1, 0, mu_B_til_ptr, 1), "culaDgemv", "mu_B_til");
  cula_exception(culaBlockDgemm('n', 'n', NA, 1, NB, 1, Z_B_ptr, NA, mu_B_ptr, NB, 0, mu_B_til_ptr, NA), "culaBlockDgemm", "mu_B_til");

  printf("computing whitened mean vector for B\n"); fflush(stdout);
/////  cula_exception(culaDgemv('t', NA, NCOM, 1, W_ptr, NA, mu_B_til_ptr, 1, 0, mu_B_til_white_ptr, 1), "culaDgemv", "mu_B_til_white"); // compute whitened mean vector B
  cula_exception(culaBlockDgemm('t', 'n', NCOM, 1, NA, 1, W_ptr, NA, mu_B_til_ptr, NA, 0, mu_B_til_white_ptr, NCOM), "culaBlockDgemm", "mu_B_white"); // compute whitened mean vector B using partitioned matrix multiplication

  printf("computing whitened mean vector for C (step 1)\n"); fflush(stdout);
/////  cula_exception(culaDgemv('n', NA, NC, 1, Z_C_ptr, NA, mu_C_ptr, 1, 0, mu_C_til_ptr, 1), "culaDgemv", "mu_C_til");
  cula_exception(culaBlockDgemm('n', 'n', NA, 1, NC, 1, Z_C_ptr, NA, mu_C_ptr, NC, 0, mu_C_til_ptr, NA), "culaBlockDgemm", "mu_C_til");

  printf("computing whitened mean vector for C\n"); fflush(stdout);
/////  cula_exception(culaDgemv('t', NA, NCOM, 1, W_ptr, NA, mu_C_til_ptr, 1, 0, mu_C_til_white_ptr, 1), "culaDgemv", "mu_C_til_white"); // compute whitened mean vector C
  cula_exception(culaBlockDgemm('t', 'n', NCOM, 1, NA, 1, W_ptr, NA, mu_C_til_ptr, NA, 0, mu_C_til_white_ptr, NCOM), "culaBlockDgemm", "mu_C_white"); // compute whitened mean vector C using partitioned matrix multiplication


  printf("pre-processing completed; ready for stochastic method\n");
  culaShutdown();
  return 0;
}
