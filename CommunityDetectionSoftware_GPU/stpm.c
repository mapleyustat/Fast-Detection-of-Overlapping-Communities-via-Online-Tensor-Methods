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


#include "stpm.h"
// this file was formerly known as TenPower_devCULA_CUDA_Whitening_MethodII

typedef unsigned long long timestamp_t;
typedef int bool_t;
enum
{
  GPU_A = 0,
  GPU_B = 1
};

enum
{
  T_TRUE = 0,
  T_FALSE = 1
};

/*
 *  Global vars and buffers
 */
struct timeval start_timeval_stpm, stop_timeval_stpm;

double Glob_Buffer_Mat_1[NCOM*NCOM];                    // evec_mat
double Glob_Buffer_Mat_2[16];                           // scaling_mat_g_C_vec_buff
double Glob_Buffer_Mat_3[16];                           // scaling_mat_mu_C_vec_buff
double Glob_Buffer_Mat_4[NCOM*NCOM];                    // eye_NCOM_buff
double Glob_Buffer_Mat_5[NCOM*NCOM];                    // for file I/O of eigenvectors
double Glob_Buffer_Mat_6[FROB_NORM_VALS_BUFF_SIZE];     // for file I/O of frob norm values
double Glob_Buffer_Mat_7[NCOM];                         // for file I/O of eigenvalues
double Glob_Buffer_Mat_8[NCOM*NCOM];                    // eigenvalue matrix for post processing

extern double G_XA_white_buff[NCOM*NX];                 // defined externally in pre_proc.c
extern double G_XB_til_white_buff[NCOM*NX];             // defined externally in pre_proc.c
extern double G_XC_til_white_buff[NCOM*NX];             // defined externally in pre_proc.c
extern double mu_A_white_buff[NCOM];                    // for mu_XA. Defined externally in pre_proc.c
extern double mu_B_til_white_buff[NCOM];                // for mu_XB. Defined externally in pre_proc.c
extern double mu_C_til_white_buff[NCOM];                // for mu_XC. Defined externally in pre_proc.c

extern int pre_proc(void);                             // for pre-processing/whitening. Defined externally in pre_proc.c
extern int post_proc(void);                            // for post-processing (computing Pi). Defined externally in post_proc.c


int selectedGPU;          // Selected GPU at command line
bool_t forceExit_flag;    // Flag to forece exit the process

extern struct timeval start_timeval_svd1, stop_timeval_svd1;
extern struct timeval start_timeval_svd2, stop_timeval_svd2;
timestamp_t measure_start_svd1, measure_stop_svd1;       // Timing for svd1
timestamp_t measure_start_svd2, measure_stop_svd2;       // Timing for svd2
timestamp_t measure_start_pre, measure_stop_pre;       // Timing for pre_proc 
timestamp_t measure_start_stpm, measure_stop_stpm;     // Timing for stpm
timestamp_t measure_start_post, measure_stop_post;     // Timing for post_proc
struct timeval start_timeval_pre, stop_timeval_pre;
struct timeval start_timeval_stpm, stop_timeval_stpm;
struct timeval start_timeval_post, stop_timeval_post;

double time_pre, time_stpm, time_post;                 // Time taken 
double time_svd1, time_svd2;

/*
 *  CUDA Kernel wrappers
 */
void vecSq_CudaKer(double *, int);
void vecInv_CudaKer(double *, int, double);
void saveFrobNormVal_CudaKer(double *, double);
void genSigmaMat_CudaKer(double *, double *, int, int);
void vecSqrt_CudaKer(double *, int);
void vecRecprocalSqrt_CudaKer(double *, int);
void l2Norm_CudaKer(double *, int, double *); // computes L2-Norm of the input vector
void pow3By2Evals_CudaKer(double *, int); // computes 3/2th power of each element of the input vector
void genInvEvalMat_CudaKer(double *, double *, int, int, double *, double); // matricizes the inverse normalized eigenvalues
void fill_iter_mat_Vals_CudaKer(double *, double *, int);

void fillVal_CudaKer(double *);

/*
 * This function checks for CULA requirements compatibility
 */
int MeetsMinimumCulaRequirements()
{
  int cudaMinimumVersion = culaGetCudaMinimumVersion();
  int cudaRuntimeVersion = culaGetCudaRuntimeVersion();
  int cudaDriverVersion = culaGetCudaDriverVersion();
  int cublasMinimumVersion = culaGetCublasMinimumVersion();
  int cublasRuntimeVersion = culaGetCublasRuntimeVersion();
  
  if(cudaRuntimeVersion < cudaMinimumVersion)
  {
    printf("CUDA runtime version is insufficient; "
	   "version %d or greater is required\n", cudaMinimumVersion);
    return 0;
  }
  
  if(cudaDriverVersion < cudaMinimumVersion)
  {
    printf("CUDA driver version is insufficient; "
	   "version %d or greater is required\n", cudaMinimumVersion);
    return 0;
  }
  
  if(cublasRuntimeVersion < cublasMinimumVersion)
  {
    printf("CUBLAS runtime version is insufficient; "
	   "version %d or greater is required\n", cublasMinimumVersion);
    return 0;
  }
  
  return 1;
}


/*
 * This is the thread routine that executes on GPU
 */
void * GPU_Task(void *arg)
{
  /****************** 
   *  Variables
   *******************/
  culaStatus status;
  culaStatus status1;
  cudaError_t err_cuda;
  
  int frob_norm_test_counter = 0;
  int fl_indx = 0;
  int idx_A = 0;
  int idx_B = 0;
  int idx_C = 0;
  
  bool_t convergence = T_FALSE;
  int indx_i = 0, indx_j = 0;
  bool_t STPMLoopTerm_flag = T_FALSE;
  
  FILE * eigVec_fptr = NULL;
  FILE * eigVal_fptr = NULL;
  FILE * frobNorm_fptr = NULL;
  FILE * miscInfo_fptr = NULL;
  
  // CPU Buffers
  double * scaling_mat_g_C_vec_buff_cpu = NULL;
  double * scaling_mat_mu_C_vec_buff_cpu =  NULL;
  double * eye_NCOM_buff_cpu = NULL;
  
  double * evec_mat = NULL;
  double evec_evec_mat_ptr_CPU_buff[NCOM];
  double *eval_vec_ptr_CPU_buff = NULL;
  
  // For file I/O
  double * frob_norm_val_buffer_cpu = NULL;
  double * eigenVector_buffer_cpu = NULL;
  
  // GPU Buffers
  double * iter_mat_buff = NULL;
  double * mu_A_vec_dev = NULL;
  double * mu_B_vec_dev = NULL;
  double * mu_C_vec_buff = NULL;
  double * g_A_vec_dev = NULL;
  double * g_B_vec_dev = NULL;
  double * g_C_vec_buff = NULL;
  double * Phi_mat_buff = NULL;
  double * evec_mat_buff = NULL;
  double * evec_mat_buff_1 = NULL;
  double * evec_mat_buff_cpy = NULL;
  double * evec_evec_mat_buff = NULL;
  double * iter_mat_interm_mem_dev = NULL;
  
  double * scaling_mat_g_C_vec_buff;  // 4x4 intermediate matrix
  double * scaling_mat_mu_C_vec_buff; // 4x4 intermediate matrix
  double * eye_NCOM_buff = NULL;
  double * vos_g_C_buff;  // vector of scalars for g_C
  double * vos_mu_C_buff; // vector of scalars for mu_C
  double * eval_vec_buff;
  
  double * frob_norm_val_buffer = NULL;
  
  int i, j; // all local variable and pointer declarations
  int sgd_iter = 0; // counter for the number of iterations of stochastic power method
  double *iter_mat_ptr = NULL, *Phi_mat_ptr = NULL, *phi_ptr = NULL;
  double *evec_mat_ptr = NULL, *evec_mat_ptr_1 = NULL, *evec_mat_ptr_cpy = NULL;
  double *evec_evec_mat_ptr = NULL; // let eigenvector matrix be E; then this is the pointer to the buffer containing E^T.E
  double *scaling_mat_g_C_vec_ptr = NULL, *scaling_mat_mu_C_vec_ptr = NULL; // 4x4 matrices hard-coded
  double tmp1, *tmp2, tmp_1 = 1, *tmp_1_ptr = NULL; // facilitating variables
  double *vos_g_C_ptr = NULL, *vos_mu_C_ptr = NULL, *tmp_ptr = NULL; // vector of scalars
  double *g_C_vec_ptr = NULL, *mu_C_vec_ptr = NULL; // vectors to be scaled (blue-purple part)
  double *eye_ptr = NULL; // identity matrix pointer
  double *eval_vec_ptr = NULL, sum_eval = 0, *sum_eval_ptr = NULL; // pointers and variabes used for eigenvalue computation
  double cvg_tst = 1, *cvg_tst_ptr = NULL; // Needed for Frobenius Norm
  double *l2Norm_val_dev = NULL;
  double *post_proc_eval_mat = NULL;
  double *eval_mat_buff_CPU = NULL;
  
  /******************
   *  fopens for fileI/O
   *******************/
  eigVec_fptr = fopen(EVECS, "w");
  if(eigVec_fptr == NULL)
  {
    fprintf(stderr,"Error opening file eigVectors.txt!!!!\n");
  }
  
  eigVal_fptr = fopen(EVALS, "w");
  if(eigVal_fptr == NULL)
  {
    fprintf(stderr,"Error opening file eigValues.txt!!!!\n");
  }
  
  frobNorm_fptr = fopen(FROB_NORM, "w");
  if(frobNorm_fptr == NULL)
  {
    fprintf(stderr,"Error opening file frobNormVals.txt!!!!\n");
  }
  
  miscInfo_fptr = fopen(SIM_INFO, "w");
  if(miscInfo_fptr == NULL)
  {
    fprintf(stderr,"Error opening file simMiscInfo.txt!!!!\n");
  }
  
  /******************
   *  CULA Initializations
   *******************/
  status = culaSelectDevice(selectedGPU); // Select which GPU to execute on
  if(status != culaNoError)
  {
    fprintf(stderr,"ERROR: BAD DEVICE ID ENTERED!!!\n");
  }
  
  status = culaInitialize();                 // connect to the GPU select above
  if(status != culaNoError)
  {
    fprintf(stderr,"ERROR: INITIALIZATION OF GPU FAILED!!!\n");
  }
    
  /******************    
   *  Buffer Allocations
   *******************/
  // For File I/O
  eigenVector_buffer_cpu = Glob_Buffer_Mat_5;
  frob_norm_val_buffer_cpu = Glob_Buffer_Mat_6;
  eval_vec_ptr_CPU_buff = Glob_Buffer_Mat_7;
  
  // CPU Buffers (Read-only)
  evec_mat = Glob_Buffer_Mat_1;
  scaling_mat_g_C_vec_buff_cpu = Glob_Buffer_Mat_2;
  scaling_mat_mu_C_vec_buff_cpu =  Glob_Buffer_Mat_3;
  eye_NCOM_buff_cpu = Glob_Buffer_Mat_4;
  
  // Buffers for Post-Proc
  eval_mat_buff_CPU = Glob_Buffer_Mat_8;
  
  // Allocate GPU memory
  err_cuda = cudaMalloc((void**)&iter_mat_buff, sizeof(double)*(4*NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR iter_mat_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&mu_C_vec_buff, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR mu_C_vec_buff!!!\n");
  }
  err_cuda = cudaMalloc((void**)&g_C_vec_buff, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR g_C_vec_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&evec_mat_buff, sizeof(double)*(NCOM)*NCOM);
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR evec_mat_buff!!!\n");
  }
  err_cuda = cudaMalloc((void**)&evec_mat_buff_1, sizeof(double)*(NCOM)*NCOM);
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR evec_mat_buff_1!!!\n");
  }
  err_cuda = cudaMalloc((void**)&evec_mat_buff_cpy, sizeof(double)*(NCOM)*NCOM);
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR evec_mat_buff_cpy!!!\n");
  }
  err_cuda = cudaMalloc((void**)&evec_evec_mat_buff, sizeof(double)*(NCOM)*NCOM);
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR evec_evec_mat_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&Phi_mat_buff, sizeof(double)*(NCOM/2)*4);
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR Phi_mat_buff!!!\n");
  }
    
  err_cuda = cudaMalloc((void**)&scaling_mat_g_C_vec_buff, sizeof(double)*(16));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR scaling_mat_g_C_vec_buff!!!\n");
  }
  err_cuda = cudaMalloc((void**)&scaling_mat_mu_C_vec_buff, sizeof(double)*(16));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR scaling_mat_mu_C_vec_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&eye_NCOM_buff, sizeof(double)*(NCOM*NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR eye_NCOM_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&vos_g_C_buff, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR vos_g_C_buff!!!\n");
  }
  err_cuda = cudaMalloc((void**)&vos_mu_C_buff, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR vos_mu_C_buff!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&eval_vec_buff, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR eval_vec_buff!!!\n");
  }
  err_cuda = cudaMalloc((void**)&tmp2, sizeof(double)*(4));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR tmp2!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&tmp_1_ptr, sizeof(double)*(1));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR tmp_1_ptr!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&cvg_tst_ptr, sizeof(double)*(1));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR cvg_tst_ptr!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&sum_eval_ptr, sizeof(double)*(1));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR sum_eval_ptr!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&frob_norm_val_buffer, sizeof(double)*(FROB_NORM_VALS_BUFF_SIZE));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR frob_norm_val_buffer!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&iter_mat_interm_mem_dev, sizeof(double)*(NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR iter_mat_interm_mem_dev!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&l2Norm_val_dev, sizeof(double)*(1));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR l2Norm_val_dev!!!\n");
  }
  
  err_cuda = cudaMalloc((void**)&post_proc_eval_mat, sizeof(double)*(NCOM*NCOM));
  if(err_cuda != cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU MALLOC FAILED FOR post_proc_eval_mat!!!\n");
  }
  
  
  // Setting GPU pointers to GPU buffers
  scaling_mat_g_C_vec_ptr = scaling_mat_g_C_vec_buff;
  scaling_mat_mu_C_vec_ptr = scaling_mat_mu_C_vec_buff;
  vos_g_C_ptr = vos_g_C_buff;
  vos_mu_C_ptr = vos_mu_C_buff;
  g_C_vec_ptr = g_C_vec_buff;
  mu_C_vec_ptr = mu_C_vec_buff;
  eye_ptr = eye_NCOM_buff;
  eval_vec_ptr = eval_vec_buff;
  tmp_ptr = tmp2;
  Phi_mat_ptr = Phi_mat_buff;
  iter_mat_ptr = iter_mat_buff;
  evec_mat_ptr = evec_mat_buff;
  evec_mat_ptr_1 = evec_mat_buff_1;
  evec_mat_ptr_cpy = evec_mat_buff_cpy;
  evec_evec_mat_ptr = evec_evec_mat_buff;
  
  /******************
   *  Host To device Memory Transfers
   *******************/
  printf("GPU: Copying Initial Eigen Vectors from Host to Device.\n");
  err_cuda = cudaMemcpy(evec_mat_buff, evec_mat, sizeof(double)*NCOM*NCOM, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY evec_mat_buff ERROR!!!\n");
  }  
  
  err_cuda = cudaMemcpy(scaling_mat_g_C_vec_buff, scaling_mat_g_C_vec_buff_cpu, sizeof(double)*16, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY scaling_mat_g_C_vec_buff ERROR!!!\n");
  }

  err_cuda = cudaMemcpy(scaling_mat_mu_C_vec_buff, scaling_mat_mu_C_vec_buff_cpu, sizeof(double)*16, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY scaling_mat_mu_C_vec_buff ERROR!!!\n");
  }
  
  err_cuda = cudaMemcpy(eye_NCOM_buff, eye_NCOM_buff_cpu, sizeof(double)*NCOM*NCOM, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eye_NCOM_buff_cpu ERROR!!!\n");
  }
  
  err_cuda = cudaMemcpy(tmp_1_ptr, &tmp_1, sizeof(double)*1, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eye_NCOM_buff_cpu ERROR!!!\n");
  }
  
  err_cuda = cudaMemcpy(sum_eval_ptr, &sum_eval, sizeof(double)*1, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY sum_val ERROR!!!\n");
  }
  
  /******************
   *  Setting pointers for 'iter_mat_buff' constituent vectors
   *******************/
  // This buffer pointed to by 'iter_mat_buff' consists of
  // the following component vectors:
  //      [mu_A_vec;mu_B_vec;g_A_vec;g_B_vec];
  // Also note that iter_mat elements will be in ColMajor format,
  mu_A_vec_dev = iter_mat_buff + 0;
  mu_B_vec_dev = iter_mat_buff + 1; // increment by 1 to perform RowMajor interleaving at appropriate location
  g_A_vec_dev  = iter_mat_buff + 2; // increment by 2 to perform RowMajor interleaving at appropriate location
  g_B_vec_dev  = iter_mat_buff + 3; // increment by 3 to perform RowMajor interleaving at appropriate location
  
  /******************
   *  Filling vectors of 'iter_mat_buff' that are independent of 'i'
   *******************/
  // Copying mu_C_vec_buff to device memory
  err_cuda = cudaMemcpy(mu_C_vec_buff, mu_C_til_white_buff, sizeof(double)*NCOM, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY mu_C_ver_buff ERROR!!!\n");
  }
  
  // Copying mu_A_vec_buff to device interm device-memory and filling in at iter_mat locations by cuda kernel call
  err_cuda = cudaMemcpy(iter_mat_interm_mem_dev, mu_A_white_buff, sizeof(double)*NCOM, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY mu_A_white_buff ERROR!!!\n");
  }
  fill_iter_mat_Vals_CudaKer(iter_mat_interm_mem_dev, mu_A_vec_dev, NCOM);
  
  // Copying mu_B_til_white_buff to device interm device-memory and filling in at iter_mat locations by cuda kernel call
  err_cuda = cudaMemcpy(iter_mat_interm_mem_dev, mu_B_til_white_buff, sizeof(double)*NCOM, cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY mu_B_til_white_buff ERROR!!!\n");
  }
  fill_iter_mat_Vals_CudaKer(iter_mat_interm_mem_dev, mu_B_vec_dev, NCOM);
  
  
  /******************
   *  Infinite loop terminating upon convergence
   *******************/
  frob_norm_test_counter = 0;
  STPMLoopTerm_flag = T_FALSE;
  gettimeofday(&start_timeval_stpm, NULL); // Measuring start time for Power Method
  while(STPMLoopTerm_flag == T_FALSE)
  {
    // Computing sample indices -- *** REMOVED RAND AND REPLACED WITH SEQUENTIAL ***
    //idx_A = rand() % NX;//NA - use this if the dataset is not randomized first
    idx_B = idx_A;
    idx_C = idx_A;
    printf("index of the selected data points: set A = %d, set B = %d, set C = %d\n", idx_A, idx_B, idx_C);
    
    //*** Extracting Vectors that depend on 'idx_A','idx_B' and 'idx_C' and filling iter_mat***//
    // Copying G_XA_white_buff's i-th column to device interm device-memory and filling in at iter_mat locations by cuda kernel call
    err_cuda = cudaMemcpy(iter_mat_interm_mem_dev, &G_XA_white_buff[idx_A*NCOM], sizeof(double)*NCOM, cudaMemcpyHostToDevice);
    if(err_cuda !=cudaSuccess)
    {
      fprintf(stderr,"ERROR: GPU CUDAMEMCPY G_XA_white_buff ERROR!!!\n");
    }
    fill_iter_mat_Vals_CudaKer(iter_mat_interm_mem_dev, g_A_vec_dev, NCOM);
    
    
    // Copying G_XB_til_white_buff's j-th column to device interm device-memory and filling in at iter_mat locations by cuda kernel call
    err_cuda = cudaMemcpy(iter_mat_interm_mem_dev, &G_XB_til_white_buff[idx_B*NCOM], sizeof(double)*NCOM, cudaMemcpyHostToDevice);
    if(err_cuda !=cudaSuccess)
    {
      fprintf(stderr,"ERROR: GPU CUDAMEMCPY G_XB_til_white_buff ERROR!!!\n");
    }
    fill_iter_mat_Vals_CudaKer(iter_mat_interm_mem_dev, g_B_vec_dev, NCOM);
    
    
    // Copying G_XC_til_white_buff's k-th column to g_C_vec_buff
    err_cuda = cudaMemcpy(g_C_vec_buff, &G_XC_til_white_buff[idx_C*NCOM], sizeof(double)*NCOM, cudaMemcpyHostToDevice);
    if(err_cuda !=cudaSuccess)
    {
      fprintf(stderr,"ERROR: GPU CUDAMEMCPY G_XC_til_white_buff ERROR!!!\n");
    }
    
    // Computing Mat-Mat mult: Phi_mat_buff = iter_mat_buff * evec_mat_buff
    status = culaDeviceDgemm('n', 'n', 4, (NCOM), NCOM, 1.0, iter_mat_ptr, 4,  evec_mat_ptr, NCOM, 0.0, Phi_mat_ptr, 4);// Phi 1/2 (for now, single GPU)
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemm 1 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
    
#ifdef DEBUG_PRINTS
    printf("Here---1\n");
    fflush(stdout);
#endif		
    for(i=0; i<NCOM; i++)
    {
      phi_ptr = Phi_mat_ptr+4*i;
      status = culaDeviceDgemv('n', 4, 4, 1, scaling_mat_g_C_vec_ptr, 4, phi_ptr, 1, 0, tmp_ptr, 1); // matrix . vector
#ifdef CULA_DEBUG_PRINTS
      if(status != culaNoError)
      {
	fprintf(stderr,"ERROR: culaDeviceDgemv 4 !!!!\n");
	fprintf(stderr,"Press any key to continue..\n");
	getchar();
      }
#endif
      
      status = culaDeviceDgemv('n', 1, 4, 1, tmp_ptr, 1, phi_ptr, 1, 0, vos_g_C_ptr+i, 1); // vector . vector
#ifdef CULA_DEBUG_PRINTS
      if(status != culaNoError)
      {
	fprintf(stderr,"ERROR: culaDeviceDgemv 5 !!!!\n");
	fprintf(stderr,"Press any key to continue..\n");
	getchar();
      }
#endif
    }
#ifdef DEBUG_PRINTS
    printf("Here---2\n");
    fflush(stdout); 
#endif	
    for(i=0; i<NCOM; i++) // BETTER TO PARALLELIZE MU AND GC? ONLY OVERHEAD GPU COMMUNICATION... think about it
    {
      phi_ptr = Phi_mat_ptr+4*i;
      status = culaDeviceDgemv('n', 4, 4, 1, scaling_mat_mu_C_vec_ptr, 4, phi_ptr, 1, 0, tmp_ptr, 1);
#ifdef CULA_DEBUG_PRINTS
      if(status != culaNoError)
      {
	fprintf(stderr,"ERROR: culaDeviceDgemv 6 !!!!\n");
	fprintf(stderr,"Press any key to continue..\n");
	getchar();
      }
#endif
      status = culaDeviceDgemv('n', 1, 4, 1, tmp_ptr, 1, phi_ptr, 1, 0, vos_mu_C_ptr+i, 1);
#ifdef CULA_DEBUG_PRINTS
      if(status != culaNoError)
      {
	fprintf(stderr,"ERROR: culaDeviceDgemv 7 !!!!\n");
	fprintf(stderr,"Press any key to continue..\n");
	getchar();
      }
#endif
    }
    
#ifdef DEBUG_PRINTS
    printf("Here---3\n");
    fflush(stdout);
#endif	
    err_cuda = cudaMemcpy(evec_mat_ptr_1, evec_mat_ptr, NCOM*NCOM*sizeof(double),cudaMemcpyDeviceToDevice); // for the blue-purple (negated) part
#ifdef CULA_DEBUG_PRINTS
    if(err_cuda !=cudaSuccess)
    {
      fprintf(stderr,"ERROR: GPU CUDAMEMCPY evec_mat_ptr_1 ERROR!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    err_cuda = cudaMemcpy(evec_mat_ptr_cpy, evec_mat_ptr, NCOM*NCOM*sizeof(double),cudaMemcpyDeviceToDevice); // for testing convergence
#ifdef CULA_DEBUG_PRINTS
    if(err_cuda !=cudaSuccess)
    {
      fprintf(stderr,"ERROR: GPU CUDAMEMCPY evec_mFROB_NORM_VALS_BUFF_SIZEat_ptr_cpy ERROR!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---4\n");
    fflush(stdout);
#endif	
    status = culaDeviceDgemm('n', 'n', NCOM, NCOM, 1, 1, g_C_vec_ptr, NCOM, vos_g_C_ptr, 1, 0, evec_mat_ptr_1, NCOM);
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemm !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
    
    status = culaDeviceDgemm('n', 'n', NCOM, NCOM, 1, 1, mu_C_vec_ptr, NCOM, vos_mu_C_ptr, 1, 1, evec_mat_ptr_1, NCOM);
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemm !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---5\n");
    fflush(stdout);
#endif	
    status = culaDeviceDgemm('t', 'n', NCOM, NCOM, NCOM, 1, evec_mat_ptr, NCOM, evec_mat_ptr, NCOM, 0, evec_evec_mat_ptr, NCOM); // note: transpose is correct
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      int info = culaGetErrorInfo();
      char buf[256];
      culaGetErrorInfoString(status, info, buf, sizeof(buf));
      printf("%s", buf);
      
      fprintf(stderr,"ERROR: culaDeviceDgemm 2 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---6\n");
    fflush(stdout);
#endif	
    vecSq_CudaKer(evec_evec_mat_ptr, NCOM*NCOM);
    cudaDeviceSynchronize();
    
#ifdef DEBUG_PRINTS
    printf("Here---7\n");
    fflush(stdout);
#endif	
    status = culaDeviceDgemm('n', 'n', NCOM, NCOM, NCOM, -LEARN_RATE, evec_mat_ptr, NCOM, evec_evec_mat_ptr, NCOM, 1, evec_mat_ptr, NCOM); 
    // eigenvector matrix updated = old eigenvector matrix - LEARN_RATE*old eigenvector matrix*...
    // [(eigenvector matrix dot eigenvector matrix).^2 -- note this is symmetric]
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemm 3 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---8\n");
    fflush(stdout);
#endif	
    status = culaDeviceDgemm('n', 'n', NCOM, NCOM, NCOM, +LEARN_RATE, eye_ptr, NCOM, evec_mat_ptr_1, NCOM, 1, evec_mat_ptr, NCOM); // stochastic gradient multipled by negative of the learning rate--the rate is positive
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemm 4 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---9\n");
    fflush(stdout);
#endif	
    status = culaDeviceDgemm('n', 'n', NCOM, NCOM, NCOM, 1, eye_ptr, NCOM, evec_mat_ptr, NCOM, -1, evec_mat_ptr_cpy, NCOM); // convergence test by frobenius norm
#ifdef CULA_DEBUG_PRINTS
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemv 5 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
#endif
    
#ifdef DEBUG_PRINTS
    printf("Here---10\n");
    fflush(stdout);
#endif	
    
    // If iterations have reached ITERS_BEFORE_CONV_TEST count, then test for convergence
    if((sgd_iter % ITERS_BEFORE_CONV_TEST)==0)
    {
      status = culaDeviceDgemv('n', 1, NCOM*NCOM, 1, evec_mat_ptr_cpy, 1, evec_mat_ptr_cpy, 1, 0, cvg_tst_ptr, 1);
#ifdef CULA_DEBUG_PRINTS
      if(status != culaNoError)
      {
	fprintf(stderr,"ERROR: culaDeviceDgemv 12 !!!!\n");
	fprintf(stderr,"Press any key to continue..\n");
	getchar();
      }
#endif
      
      err_cuda = cudaMemcpy(&cvg_tst, cvg_tst_ptr, sizeof(double)*1, cudaMemcpyDeviceToHost);
      if(err_cuda !=cudaSuccess)
      {
	fprintf(stderr,"ERROR: GPU CUDAMEMCPY cvg_tst ERROR!!!\n");
      }
      
      printf("(Iter %d) Frob Norm = %5.25e\n", sgd_iter, cvg_tst);
      fflush(stdout);
      
      // Writing current Frob Norm value to buffer for writing in file
      saveFrobNormVal_CudaKer(&frob_norm_val_buffer[frob_norm_test_counter],cvg_tst);
      frob_norm_test_counter++;
      
      // Check if convergence has been achieved or not
      if((cvg_tst < (double)THRESH) && (sgd_iter > (double)SGD_ITER_MIN))
      {
	printf("Convergence Achieved\n");
	STPMLoopTerm_flag = T_TRUE;
      }
      
      // Exit if enforced by user
      if(forceExit_flag == T_TRUE)
      {
	printf("Exiting STPM loop now\n");
	STPMLoopTerm_flag = T_TRUE;
      }
      
      // Exit if maximum number of iterations reached!
      if(sgd_iter > SGD_ITER_MAX)
      {
	printf("Maximum number of iterations reached! Exiting!\n");
	STPMLoopTerm_flag = T_TRUE;
      }
    }
    
    idx_A = (idx_A+1)%(NX); // NX+1 in previous version of the code; modified to NX to be safe
    sgd_iter++;




//[ this block was added as a normalization / stabilization for dblp to prevent nan frob_norm values; comment this block to obtain the previous version of the code used for yelp, facebook
    double *dev_reflectors = NULL;
    
    err_cuda = cudaMalloc((void**)&dev_reflectors, NCOM*sizeof(double));
    if(err_cuda != cudaSuccess)
    {
      fprintf(stderr,"&&&&&&&&&& gpu malloc for qr for device evecs failed!!!\n");
    }
    
    culaDeviceDgeqrfp(NCOM, NCOM, evec_mat_ptr, NCOM, dev_reflectors); // Compute QR for initialization
    culaDeviceDorgqr(NCOM, NCOM, NCOM, evec_mat_ptr, NCOM, dev_reflectors); // Generate Q
    cudaFree(dev_reflectors);
//]



  }
  
  
  /******************
   *  Computing eigenvalues
   *******************/
  printf("Computing Eigenvalues\n");
  for(i=0; i<NCOM; i++)
  {
    status = culaDeviceDgemv('t', NCOM, 1, 1, evec_mat_ptr+(NCOM*i), NCOM, evec_mat_ptr+(NCOM*i), 1, 0, eval_vec_ptr+i, 1); // note: transpose is correct
    if(status != culaNoError)
    {
      fprintf(stderr,"ERROR: culaDeviceDgemv 13 !!!!\n");
      fprintf(stderr,"Press any key to continue..\n");
      getchar();
    }
  }
  
  // NOTE: At this point eigenvalues have been modified to 3/2th power 
  pow3By2Evals_CudaKer(eval_vec_ptr, NCOM);
  
  gettimeofday(&stop_timeval_stpm, NULL);  // Measuring stop time for Power Method
  
  /******************    
   *  Post processing
   *******************/
  printf("\npost-processing\n");
  fflush(stdout);
  
  err_cuda = cudaMemcpy(eigenVector_buffer_cpu, evec_mat_ptr, NCOM*NCOM*sizeof(double),cudaMemcpyDeviceToHost);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eigenVector_buffer_cpu ERROR!!!\n");
  }
  
  gettimeofday(&start_timeval_post, NULL);  // Measuring start time for post processing
  
  l2Norm_CudaKer(eval_vec_ptr, NCOM, l2Norm_val_dev);
  genInvEvalMat_CudaKer(eval_vec_ptr, post_proc_eval_mat, NCOM, NCOM, l2Norm_val_dev, (double)PINV_TOL);
  
  err_cuda = cudaMemcpy(eval_mat_buff_CPU, post_proc_eval_mat, NCOM*NCOM*sizeof(double),cudaMemcpyDeviceToHost);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eval_mat_buff_CPU ERROR!!!\n");
    fprintf(stderr,"Press any key to continue..\n");
    getchar();
  }
  
  err_cuda = cudaMemcpy(eval_vec_ptr_CPU_buff, eval_vec_ptr, NCOM*sizeof(double),cudaMemcpyDeviceToHost);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eval_vec_ptr_CPU_buff ERROR!!!\n");
    fprintf(stderr,"Press any key to continue..\n");
    getchar();
  }
  
  post_proc(); 
  
  for(i=0; i<NCOM; i++) // calculating elementwise inverse squared
  {
    *(eval_vec_ptr_CPU_buff+i) = 1/(*(eval_vec_ptr_CPU_buff+i) * *(eval_vec_ptr_CPU_buff+i));
    sum_eval += *(eval_vec_ptr_CPU_buff+i);
  }
  sum_eval = 1/sum_eval;
  err_cuda = cudaMemcpy(eval_vec_ptr, eval_vec_ptr_CPU_buff, NCOM*sizeof(double),cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eval_vec_ptr ERROR!!!\n");
    fprintf(stderr,"Press any key to continue..\n");
    getchar();
  }
  
  err_cuda = cudaMemcpy(sum_eval_ptr, &sum_eval, 1*sizeof(double),cudaMemcpyHostToDevice);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eval_vec_ptr_CPU_buff ERROR!!!\n");
    fprintf(stderr,"Press any key to continue..\n");
    getchar();
  }
  
  status = culaDeviceDgemv('n', NCOM, 1, 1, eval_vec_ptr, NCOM, sum_eval_ptr, 1, 0, eval_vec_ptr, 1); // normalizing for dirichlet
  if(status != culaNoError)
  {
    fprintf(stderr,"ERROR: culaDeviceDgemv 14!!!!\n");
    fprintf(stderr,"Press any key to continue..\n");
    getchar();
  }
  
  gettimeofday(&stop_timeval_post, NULL);  // Measuring stop time for post processing
  
  /****************** 
   *  Total Exec time analysis
   *******************/
  // Pre-processing SVD1 timing
  measure_stop_svd1 = stop_timeval_svd1.tv_usec + (timestamp_t)stop_timeval_svd1.tv_sec * 1000000;
  measure_start_svd1 = start_timeval_svd1.tv_usec + (timestamp_t)start_timeval_svd1.tv_sec * 1000000;
  time_svd1 = (measure_stop_svd1 - measure_start_svd1) / 1000000.0L;
  printf("Exec Time svd1 = %5.25e (Seconds)\n", time_svd1);

  // Pre-processing SVD2 timing
  measure_stop_svd2 = stop_timeval_svd2.tv_usec + (timestamp_t)stop_timeval_svd2.tv_sec * 1000000;
  measure_start_svd2 = start_timeval_svd2.tv_usec + (timestamp_t)start_timeval_svd2.tv_sec * 1000000;
  time_svd2 = (measure_stop_svd2 - measure_start_svd2) / 1000000.0L;
  printf("Exec Time svd1 = %5.25e (Seconds)\n", time_svd2);

  // Pre-processing timing
  measure_stop_pre = stop_timeval_pre.tv_usec + (timestamp_t)stop_timeval_pre.tv_sec * 1000000;
  measure_start_pre = start_timeval_pre.tv_usec + (timestamp_t)start_timeval_pre.tv_sec * 1000000;
  time_pre = (measure_stop_pre - measure_start_pre) / 1000000.0L;
  printf("Exec Time Pre Proc = %5.25e (Seconds)\n",time_pre);
  
  // STPM timing   
  measure_stop_stpm = stop_timeval_stpm.tv_usec + (timestamp_t)stop_timeval_stpm.tv_sec * 1000000;
  measure_start_stpm = start_timeval_stpm.tv_usec + (timestamp_t)start_timeval_stpm.tv_sec * 1000000;
  time_stpm = (measure_stop_stpm - measure_start_stpm) / 1000000.0L;
  printf("Exec Time STPM = %5.25e (Seconds)\n",time_stpm);
  
  // Post-processing timing
  measure_stop_post = stop_timeval_post.tv_usec + (timestamp_t)stop_timeval_post.tv_sec * 1000000;
  measure_start_post = start_timeval_post.tv_usec + (timestamp_t)start_timeval_post.tv_sec * 1000000;
  time_post = (measure_stop_post - measure_start_post) / 1000000.0L;
  printf("Exec Time Post Proc = %5.25e (Seconds)\n",time_post);
  
  printf("\nTotal Exec Time = %5.25e (Seconds)\n",time_pre + time_stpm + time_post);
  
  /******************    
   *  Writing results to files
   *******************/
  // Writing eigenvalues to file
  printf("Writing eigenvalues to file\n");
  fflush(stdout);
  err_cuda = cudaMemcpy(eval_vec_ptr_CPU_buff, eval_vec_ptr, NCOM*sizeof(double),cudaMemcpyDeviceToHost);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY eval_vec_ptr_CPU_buff ERROR!!!\n");
  }
  for(fl_indx = 0; fl_indx < NCOM; fl_indx++)
  {
    fprintf(eigVal_fptr, "%5.25e\n",eval_vec_ptr_CPU_buff[fl_indx]);
  }
  
  
  // Writing eigenvectors to file
  printf("Writing eigenvectors to file\n");
  fflush(stdout);
  for(fl_indx = 0; fl_indx < NCOM*NCOM; fl_indx++)
  {
    fprintf(eigVec_fptr, "%5.25e\n",eigenVector_buffer_cpu[fl_indx]);
  }
  
  // Writing frob norm values to file
  printf("Writing frob norm values to file\n");
  fflush(stdout);
  err_cuda = cudaMemcpy(frob_norm_val_buffer_cpu, frob_norm_val_buffer, frob_norm_test_counter*sizeof(double),cudaMemcpyDeviceToHost);
  if(err_cuda !=cudaSuccess)
  {
    fprintf(stderr,"ERROR: GPU CUDAMEMCPY frob_norm_val_buffer_cpu ERROR!!!\n");
  }
  for(fl_indx = 0; fl_indx < frob_norm_test_counter; fl_indx++)
  {
    fprintf(frobNorm_fptr, "%5.25e\n",frob_norm_val_buffer_cpu[fl_indx]);
  }
  
  
  // Writing simulation misc info to file
  printf("Writing simulation misc info to file\n");
  fflush(stdout);
  fprintf(miscInfo_fptr, "Total number of nodes          = %d\n", (int)NX+(int)NA+(int)NB+(int)NC);
  fprintf(miscInfo_fptr, "Number of nodes in partition X = %d\n", (int)NX);
  fprintf(miscInfo_fptr, "Number of nodes in partition A = %d\n", (int)NA);
  fprintf(miscInfo_fptr, "Number of nodes in partition B = %d\n", (int)NB);
  fprintf(miscInfo_fptr, "Number of nodes in partition C = %d\n", (int)NC);
  fprintf(miscInfo_fptr, "Number of nodes communities    = %d\n", (int)NCOM);
  fprintf(miscInfo_fptr, "Alpha_0 (Dirichlet parameter)  = %5.25e\n", (double)ALPHA_0);
  fprintf(miscInfo_fptr, "Learning rate = %5.25e\n", (double)LEARN_RATE);
  fprintf(miscInfo_fptr, "Threshold for STPM convergence    = %5.25e\n", (double)THRESH);
  fprintf(miscInfo_fptr, "Pseudoinverse numerical tolerance = %5.25e\n", (double)PINV_TOL);
  fprintf(miscInfo_fptr, "Total number of Frob Norm values written to file = %d\n", (int)frob_norm_test_counter);
  fprintf(miscInfo_fptr, "Number of iterations between convergence test    = %d\n", (int)ITERS_BEFORE_CONV_TEST);
  fprintf(miscInfo_fptr, "Maximum number of stochastic iterations allowed  = %d\n", (int)SGD_ITER_MAX);
  fprintf(miscInfo_fptr, "Exec Time svd1  = %5.25e (Seconds)\n", time_svd1);
  fprintf(miscInfo_fptr, "Exec Time svd2  = %5.25e (Seconds)\n", time_svd2);
  fprintf(miscInfo_fptr, "Exec Time Pre Proc  = %5.25e (Seconds)\n", time_pre);
  fprintf(miscInfo_fptr, "Exec Time STPM      = %5.25e (Seconds)\n", time_stpm);
  fprintf(miscInfo_fptr, "Exec Time Post Proc = %5.25e (Seconds)\n", time_post);
  fprintf(miscInfo_fptr, "Total Exec Time     = %5.25e (Seconds)\n", time_pre + time_stpm + time_post);
  
  /******************    
   *  Print exit message
   *******************/
  printf("\n\nTask completed.\nPress'e' (with ENTER) to exit\n");
  fflush(stdout);
  
  /******************
   *  GPU disconnect, memory free and fclose
   *******************/
  cudaFree(iter_mat_buff);
  cudaFree(mu_C_vec_buff);
  cudaFree(g_C_vec_buff);
  cudaFree(evec_mat_buff);
  cudaFree(evec_mat_buff_1);
  cudaFree(evec_mat_buff_cpy);
  cudaFree(evec_evec_mat_buff);
  cudaFree(Phi_mat_buff);
  cudaFree(scaling_mat_g_C_vec_buff);
  cudaFree(scaling_mat_mu_C_vec_buff);
  cudaFree(eye_NCOM_buff);
  cudaFree(vos_g_C_buff);
  cudaFree(vos_mu_C_buff);
  cudaFree(eval_vec_buff);
  cudaFree(tmp2);
  cudaFree(tmp_1_ptr);
  cudaFree(cvg_tst_ptr);
  cudaFree(sum_eval_ptr);
  cudaFree(frob_norm_val_buffer);
  cudaFree(iter_mat_interm_mem_dev);
  cudaFree(l2Norm_val_dev);
  cudaFree(post_proc_eval_mat);
  
  fclose(eigVec_fptr);
  fclose(eigVal_fptr);
  fclose(frobNorm_fptr);
  fclose(miscInfo_fptr);
  
  culaShutdown();  // Shutting down connection with GPU    
}

void cmd_line_usage(void)
{
  printf("usage: ./algo <option>\n");
  printf("choices for <option>:\n");
  printf("\t1) GPU_A\n");
  printf("\t2) GPU_B\n");
}


void init_buff() // initializes the 4x4 intermediate matrices and NCOMxNCOM identity matrix
{
  int i;
  
  *(Glob_Buffer_Mat_2 + 3) = -ALPHA_0/(ALPHA_0+2);
  *(Glob_Buffer_Mat_2 + 9) = -ALPHA_0/(ALPHA_0+2);
  *(Glob_Buffer_Mat_2 + 11) = 1;
  *(Glob_Buffer_Mat_3 + 1) = 2*ALPHA_0*ALPHA_0/((ALPHA_0+1)*(ALPHA_0+2));
  *(Glob_Buffer_Mat_3 + 11) = -ALPHA_0/(ALPHA_0+2);
  
  for(i=0; i<NCOM*NCOM; i+=(NCOM+1))
    Glob_Buffer_Mat_4[i]= 1; // eye_NCOM_buff
}


void init_evec_mat_buff()
{
  culaInitialize();
  int i;
  double *evec_mat_ptr = Glob_Buffer_Mat_1;
  double *reflectors = NULL;

  reflectors = (double *)malloc(NCOM*sizeof(double)); // Glob_Buffer_Mat_1 contains evec_mat_buff
  if(reflectors == NULL)
  {
    fprintf(stderr,"Error: reflectors malloc failed!!\n");
  }
  
  for(i=0; i<NCOM*NCOM; i++)
  {
    Glob_Buffer_Mat_1[i] = rand();
  }
  
  culaDgeqrfp(NCOM, NCOM, evec_mat_ptr, NCOM, reflectors); // Compute QR for initialization
  culaDorgqr(NCOM, NCOM, NCOM, evec_mat_ptr, NCOM, reflectors); // Generate Q
  free(reflectors);
  
  culaShutdown();
}



int main(int argc, char *argv[])
{
  /****************** 
   *  Variables
   *******************/
  printf("entering main\n");
  pthread_t Thrd_GPU;
  int device_count = 0;
  char str_buf[STR_BUF_SIZE];
  culaStatus status;
  
  char user_sel = 'n';
  bool_t mainTerm_flag = T_FALSE;
  
  double * scaling_mat_g_C_vec_buff_cpu = NULL;
  double * scaling_mat_mu_C_vec_buff_cpu = NULL;
  double * eye_NCOM_buff_cpu = NULL;
  double * evec_mat = NULL;
  
  /******************
   *  Initializing global flags
   *******************/
  forceExit_flag = T_FALSE;
  
  /******************
   *  argv argc processing
   *******************/
  if(argc < 2)
  {
    printf("insufficent arguments to run the program (specify the gpu to be used)\n");
    cmd_line_usage();
    printf("exited\n");
    return(0);
  }
  if(!((strcmp(argv[1],"GPU_A")==0)||(strcmp(argv[1],"GPU_B")==0)))
  {
    printf("unrecognized arguments\n");
    cmd_line_usage();
    printf("exited\n");
    return(0);
  }
  
  // Selecting GPU for the process based on cmd line input
  if(strcmp(argv[1],"GPU_A")==0)
  {
    selectedGPU = GPU_A;
  }
  else if(strcmp(argv[1],"GPU_B")==0)
  {
    selectedGPU = GPU_B;
  }
  
  /******************
   *  Setting random seed
   *******************/
  srand(time(NULL)); // use this if the dataset is not randomized first
  
  /******************
   *  Buffer Initializations
   *******************/
  evec_mat = Glob_Buffer_Mat_1;
  
  scaling_mat_g_C_vec_buff_cpu = Glob_Buffer_Mat_2;
  scaling_mat_mu_C_vec_buff_cpu =  Glob_Buffer_Mat_3;
  eye_NCOM_buff_cpu = Glob_Buffer_Mat_4;
  init_buff();
  init_evec_mat_buff();
  
  /******************     
   *  GPU Initializations and checks
   *******************/
  culaGetDeviceCount(&device_count); // get gpu count
  printf("total number of gpu's in the system = %d\n", device_count);
  
  switch(selectedGPU)
  {
  case GPU_A:
    culaGetDeviceInfo(GPU_A, str_buf, STR_BUF_SIZE); // get gpu info
    printf("GPU_A: %s\n",str_buf);
    fflush(stdout);
    break;
    
  case GPU_B:
    culaGetDeviceInfo(GPU_B, str_buf, STR_BUF_SIZE); // get gpu info
    printf("GPU_B: %s\n", str_buf);
    fflush(stdout);
    break;
    
  default:
    break;
  }
  
  if(!MeetsMinimumCulaRequirements())
  {
    printf("cula requirements not met!!\n");
  }
  
  /******************
   *  Whitening
   *******************/
  printf("pre-processing (loading datasets and then whitening) started\n");
  fflush(stdout);
  
  gettimeofday(&start_timeval_pre, NULL);  // Measuring start time for pre processing
  pre_proc();
  gettimeofday(&stop_timeval_pre, NULL);   // Measuring stop time for pre processing
  
  printf("pre-processing completed\n");
  fflush(stdout);
  
  /******************
   *  Create thread for the GPU
   *******************/	
  if(pthread_create(&Thrd_GPU,NULL,&GPU_Task,NULL))
  {
    fprintf(stderr, "thread for gpu failed!\n");
  }
  
  /******************
   *  GPU loop control
   *******************/
  mainTerm_flag = T_FALSE;
  /*
  while(mainTerm_flag != T_TRUE) // comment this loop while automating using bash script
  {
    printf("\nPress 'e' (with ENTER) at any time to exit loop\n\n");
    user_sel = getchar();
    switch(user_sel)
    {
    case 'e':
      forceExit_flag = T_TRUE;
      mainTerm_flag = T_TRUE;
      break;
      
    default:
      break;
    }
  }
  */
  
  /****************** 
   *  Waiting for thread join
   *******************/	
  pthread_join(Thrd_GPU, NULL);
  
  
  printf("end of main() from stpm.c\n");
  fflush(stdout);
  
  return 0;
}
