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

// important lesson learnt / reminded in using malloc for numerical linear algebra while coding up and debugging this file: always memset to zero while malloc'ing a pointer locally and reusing it
#include "stpm.h"
#include "curand_rand_mat.cu"


// externs from pre_proc.c
extern void cula_exception(culaStatus, const char *, const char *); // error status, cula function that fails, term in the algorithm (from pre_proc.c)


// BLAS routine written for an out-of-GPU-core solution, i.e., CPU-GPU hybrid GEMM multiplication
culaStatus culaBlockDgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) // the matrix c is cumulatively added using beta=1; so c must be initialized to zeros by programmer before calling this blocked dgemm; the input value of beta is used only the very first time beta to retain and add to the existing values in c
{
  int iblk, jblk, kblk; // loop variables for blocks
  printf("partitioned matrix multiplication using BLOCK_SIZE = %d\n", BLOCK_SIZE);

  if(transa=='n' && transb=='n')
  {
    printf("handling the case transa = n and transb = n\n"); fflush(stdout); // use loop unrolling in gcc or openmp
    for(iblk=0; iblk<=m/BLOCK_SIZE; iblk++) // note: m is int and BLOCK_SIZE is #defined as int, so division gives the int part
    {
      for(jblk=0; jblk<=n/BLOCK_SIZE; jblk++)
      {
	for(kblk=0; kblk<=k/BLOCK_SIZE; kblk++)
	{
	  int small_m=BLOCK_SIZE, small_n=BLOCK_SIZE, small_k=BLOCK_SIZE; // note: has to be declared here, not outside loops because it needs to be conditionally updated
	  if(iblk == m/BLOCK_SIZE)
	    small_m = m%BLOCK_SIZE;
	  if(jblk == n/BLOCK_SIZE)
	    small_n = n%BLOCK_SIZE;
	  if(kblk == k/BLOCK_SIZE)
	    small_k = k%BLOCK_SIZE;
	  if( (small_m==0) || (small_n==0) || (small_k==0) ) // to check for 0 of the 3 leading dimensions that results in illegal value
	    continue;
///// i j k and m n k are diff but k is common, so interpret from context - careful
	  if(kblk == 0)
	    cula_exception(culaDgemm('n', 'n', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*kblk*lda), lda, b+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*jblk*ldb), ldb, beta, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "nn");
	  else
	    cula_exception(culaDgemm('n', 'n', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*kblk*lda), lda, b+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*jblk*ldb), ldb, 1, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "nn");
	}
      }
    }
  }

  if(transa=='n' && transb=='t')
  {
    printf("handling the case transa = n and transb = t\n"); fflush(stdout); // use loop unrolling in gcc or openmp
    for(iblk=0; iblk<=m/BLOCK_SIZE; iblk++) // note: m is int and BLOCK_SIZE is #defined as int, so division gives the int part
    {
      for(jblk=0; jblk<=n/BLOCK_SIZE; jblk++)
      {
	for(kblk=0; kblk<=k/BLOCK_SIZE; kblk++)
	{
	  int small_m=BLOCK_SIZE, small_n=BLOCK_SIZE, small_k=BLOCK_SIZE; // note: has to be declared here, not outside loops because it needs to be conditionally updated
	  if(iblk == m/BLOCK_SIZE)
	    small_m = m%BLOCK_SIZE;
	  if(jblk == n/BLOCK_SIZE)
	    small_n = n%BLOCK_SIZE;
	  if(kblk == k/BLOCK_SIZE)
	    small_k = k%BLOCK_SIZE;
	  if( (small_m==0) || (small_n==0) || (small_k==0) ) // to check for 0 of the 3 leading dimensions that results in illegal value
	    continue;
///// i j k and m n k are diff but k is common, so interpret from context - careful
	  if(kblk == 0)
	    cula_exception(culaDgemm('n', 't', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*kblk*lda), lda, b+(BLOCK_SIZE*jblk)+(BLOCK_SIZE*kblk*ldb), ldb, beta, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "nt");
	  else
	    cula_exception(culaDgemm('n', 't', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*kblk*lda), lda, b+(BLOCK_SIZE*jblk)+(BLOCK_SIZE*kblk*ldb), ldb, 1, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "nt");
	}
      }
    }
  }

  if(transa=='t' && transb=='n')
  {
    printf("handling the case transa = t and transb = n\n"); fflush(stdout); // use loop unrolling in gcc or openmp
    for(iblk=0; iblk<=m/BLOCK_SIZE; iblk++) // note: m is int and BLOCK_SIZE is #defined as int, so division gives the int part
    {
      for(jblk=0; jblk<=n/BLOCK_SIZE; jblk++)
      {
	for(kblk=0; kblk<=k/BLOCK_SIZE; kblk++)
	{
	  int small_m=BLOCK_SIZE, small_n=BLOCK_SIZE, small_k=BLOCK_SIZE; // note: has to be declared here, not outside loops because it needs to be conditionally updated
	  if(iblk == m/BLOCK_SIZE)
	    small_m = m%BLOCK_SIZE;
	  if(jblk == n/BLOCK_SIZE)
	    small_n = n%BLOCK_SIZE;
	  if(kblk == k/BLOCK_SIZE)
	    small_k = k%BLOCK_SIZE;
	  if( (small_m==0) || (small_n==0) || (small_k==0) ) // to check for 0 of the 3 leading dimensions that results in illegal value
	    continue;
///// i j k and m n k are diff but k is common, so interpret from context - careful
	  if(kblk == 0)
	    cula_exception(culaDgemm('t', 'n', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*iblk*lda), lda, b+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*jblk*ldb), ldb, beta, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "tn"); // k k for a
	  else
	    cula_exception(culaDgemm('t', 'n', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*iblk*lda), lda, b+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*jblk*ldb), ldb, 1, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "tn"); // k k for a
	}
      }
    }
  }

  if(transa=='t' && transb=='t')
  {
    printf("handling the case transa = t and transb = t\n"); fflush(stdout); // use loop unrolling in gcc or openmp
    for(iblk=0; iblk<=m/BLOCK_SIZE; iblk++) // note: m is int and BLOCK_SIZE is #defined as int, so division gives the int part
    {
      for(jblk=0; jblk<=n/BLOCK_SIZE; jblk++)
      {
	for(kblk=0; kblk<=k/BLOCK_SIZE; kblk++)
	{
	  int small_m=BLOCK_SIZE, small_n=BLOCK_SIZE, small_k=BLOCK_SIZE; // note: has to be declared here, not outside loops because it needs to be conditionally updated
	  if(iblk == m/BLOCK_SIZE)
	    small_m = m%BLOCK_SIZE;
	  if(jblk == n/BLOCK_SIZE)
	    small_n = n%BLOCK_SIZE;
	  if(kblk == k/BLOCK_SIZE)
	    small_k = k%BLOCK_SIZE;
	  if( (small_m==0) || (small_n==0) || (small_k==0) ) // to check for 0 of the 3 leading dimensions that results in illegal value
	    continue;
///// i j k and m n k are diff but k is common, so interpret from context - careful
	  if(kblk == 0)
	    cula_exception(culaDgemm('t', 't', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*iblk*lda), lda, b+(BLOCK_SIZE*jblk)+(BLOCK_SIZE*kblk*ldb), ldb, beta, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "tt");
	  else
	    cula_exception(culaDgemm('t', 't', small_m, small_n, small_k, alpha, a+(BLOCK_SIZE*kblk)+(BLOCK_SIZE*iblk*lda), lda, b+(BLOCK_SIZE*jblk)+(BLOCK_SIZE*kblk*ldb), ldb, 1, c+(BLOCK_SIZE*iblk)+(BLOCK_SIZE*jblk*ldc), ldc), "culaBlockDgemm", "tt");
	}
      }
    }
  }

  return culaNoError;
}


culaStatus pinv(int m, int n, double *a, double *b) // iterative psedoinverse of ben-israel and cohen: a is the input matrix (m \times n), b is the output matrix (n \times m)
{
  int i; // for pseudo-inverse iterations
  double *temp1 = malloc(sizeof(double)*n*n); // Ai*A - n \times n
  if(temp1 == NULL) // actually malloc failure but retured as a cula error type (for convenience, so that cula_exception function may be used to catch the exception and clean-up the device)
  {
    printf("malloc failed at temp1\n");
    return culaDataError;
  }
  double *temp2 = malloc(sizeof(double)*n*m); // (Ai*A)*Ai - n \times m
  if(temp2 == NULL)
  {
    printf("malloc failed at temp2\n");
    return culaDataError;
  }
  double *bcpy = malloc(sizeof(double)*n*m); // (Ai*A)*Ai - n \times m
  if(bcpy == NULL)
  {
    printf("malloc failed at bcpy\n");
    return culaDataError;
  }
  double *temp_eye = malloc(sizeof(double)*m*m); // m \times m temporary identity matrix
  if(temp_eye == NULL)
  {
    printf("malloc failed at temp_eye\n");
    return culaDataError;
  }
  double pinv_dummy = 100;
  double *pinv_cvg_tst = &pinv_dummy;
  memset(temp_eye, 0, sizeof(double)*m*m);
  for(i=0; i<m*m; i+=m+1)
    temp_eye[i] = 1;

  cula_exception(culaDgeTranspose(m, n, a, m, b, n), "culaDgeTranspose", "pseudo-inverse initialization"); // computing out-of-place transpose of input matrix to initialize output matrix
  cula_exception(culaBlockDgemm('n', 'n', n, m, m, PINV_INIT_COEFF, b, n, temp_eye, m, 0, b, n), "culaBlockDgemm", "pseudo-inverse initialization");
  for(i=0; i<MAX_PINV_ITER; i++)
  {
    printf("pinv iter no = %d\n", i); // dgemm or block dgemm, either may be used here
    memcpy(bcpy, b, sizeof(double)*n*m);
    cula_exception(culaBlockDgemm('n', 'n', n, n, m, 1, b, n, a, m, 0, temp1, n), "culaBlockDgemm", "AiA");
    cula_exception(culaBlockDgemm('n', 'n', n, m, n, 1, temp1, n, b, n, 0, temp2, n), "culaBlockDgemm", "AiAAi");
    cula_exception(culaBlockDgemm('n', 'n', n, m, m, -1, temp2, n, temp_eye, m, 2, b, n), "culaBlockDgemm", "2Ai-AiAAi");
    cula_exception(culaBlockDgemm('n', 'n', n, m, m, -1, b, n, temp_eye, m, 1, bcpy, n), "culaBlockDgemm", "b-bcpy for convergence of pinv");
    //cula_exception(culaDgemv('n', 1, n*m, 1, bcpy, 1, bcpy, 1, 0, pinv_cvg_tst, 1), "culaDgemv", "convergence test"); // convergence test, i.e., frobenius norm implemented as an vector-vector inner product
    cula_exception(culaBlockDgemm('n', 'n', 1, 1, n*m, 1, bcpy, 1, bcpy, n*m, 0, pinv_cvg_tst, 1), "culaBlockDgemm", "bcpy*bcpy for convergence of pinv"); // replacing dgemv in the previous line by blockdgemm for out-of-gpu-core convergence check; comment or condition as needed
    if( (i>MIN_PINV_ITER) && (*pinv_cvg_tst<PINV_CVG_TST) )
    {
      printf("convergence achieved for pseudoinverse\n");
      break;
    }
  }
  free(temp1);
  free(temp2);
  free(temp_eye);

  return culaNoError;
}


culaStatus nystrom_whitening(int n, int k, double *a, double *white) // compute the whitening matrix w of a symmetric n \times n matrix a s.t. m(a,a)=I using the nystrom randomized method for k-svd based on qr of an n \times k matrix; but note that if the input matrix is not symmetric, then do 1/2(a+a') and then pass the resultant to this function
{
  printf("entered nystrom whitening function\n");
  int i; // loop counter for making singular value matrices

  double *sval = malloc(sizeof(double)*k); // vector of singular values
  if(sval == NULL)
  {
    printf("malloc failed at sval\n");
    return culaDataError;
  }
  memset(sval, 0, sizeof(double)*k);
  double *smat = malloc(sizeof(double)*k*k); // diagonal matrix of singular values
  if(smat == NULL)
  {
    printf("malloc failed at smat\n");
    return culaDataError;
  }
  memset(smat, 0, sizeof(double)*k*k);
  double *lsmat = malloc(sizeof(double)*k*k); // matrix of left sigular vectors
  if(lsmat == NULL)
  {
    printf("malloc failed at lsmat\n");
    return culaDataError;
  }
  memset(lsmat, 0, sizeof(double)*k*k);
  double *rsmat = malloc(sizeof(double)*k*k); // matrix of right singular vectors
  if(rsmat == NULL)
  {
    printf("malloc failed at rsmat\n");
    return culaDataError;
  }
  memset(rsmat, 0, sizeof(double)*k*k);


  double *s = malloc(sizeof(double)*n*k); // random matrix with entries being iid sampled from N(0, 1) generated on device using curand
  if(s == NULL)
  {
    printf("malloc failed at s\n");
    return culaDataError;
  }
  memset(s, 0, sizeof(double)*n*k);
  double *c = malloc(sizeof(double)*n*k); // C = A*S; refer mahoney's paper or survey
  if(c == NULL)
  {
    printf("malloc failed at c\n");
    return culaDataError;
  }
  memset(c, 0, sizeof(double)*n*k);
  double *r = malloc(sizeof(double)*k*k); // R from QR decomposition; its pseudoinverse is also overwritten in this
  if(r == NULL)
  {
    printf("malloc failed at r\n");
    return culaDataError;
  }
  memset(r, 0, sizeof(double)*k*k);
  double *w = malloc(sizeof(double)*k*k); // w matrix (this is not the whitening matrix) in nystrom method
  if(w == NULL)
  {
    printf("malloc failed at w\n");
    return culaDataError;
  }
  memset(w, 0, sizeof(double)*k*k);


  double *reflectors = malloc(sizeof(double)*k); // supporting variable to store the reflectors for QR
  if(reflectors == NULL)
  {
    printf("malloc failed at reflectors\n");
    return culaDataError;
  }
  memset(reflectors, 0, sizeof(double)*k);

  gen_rnd_mat(s, n*k, 0, 0.1); // s, i.e., generate the gaussian random matrix S - mean and variance are the last 2 parameters

  cula_exception(culaBlockDgemm('n', 'n', n, k, n, 1, a, n, s, n, 0, c, n), "culaBlockDgemm", "C = A*S, i.e., multiplying by random matrix");
  cula_exception(culaBlockDgemm('t', 'n', k, k, n, 1, s, n, c, n, 0, w, k), "culaBlockDgemm", "W = S'*A*S = S'*C");
  cula_exception(culaDgeqrfp(n, k, a, n, reflectors), "culaDgeqrfp", "qr step 1 in nystrom");
  cula_exception(culaDlacpy('U', k, k, a, n, r, k), "culaDlacpy", "copying r");
  cula_exception(culaDorgqr(n, k, k, a, n, reflectors), "culaDorgqr", "qr step 2 in nystrom"); // Generate Q and reuse a for n \times k matrix

  cula_exception(culaDgesvd('A', 'A', k, k, r, k, sval, lsmat, k, rsmat, k), "culaDgesvd", "for pinv(R)");
  for(i=0; i<k; i++) // inverting the vector of singular values elementwise and copying onto the diagonal sigular value matrix
    if(fabs(*(sval+i)) > PINV_TOL)
      *(smat+(k*i+i)) = 1/(*(sval+i));
  // computing pseudoinverse of k \times k R via svd and 2 matrix mutiplications; alternatively, ben-israel and cohen iterations might be used, i.e., V * sig.^-1 * U^T; reuse r for pinv(r)
  cula_exception(culaBlockDgemm('t', 'n', k, k, k, 1, rsmat, k, smat, k, 0, r, k), "culaBlockDgemm", "first multiplication for r");
  cula_exception(culaBlockDgemm('n', 't', k, k, k, 1, r, k, lsmat, k, 0, r, k), "culaBlockDgemm", "second multiplication for r");

  for(i=0; i<k*k; i++)
    if(fabs(*(r+i)) < PINV_TOL) // small value errors (clamp non-zero matrix entries to zero if too small)
      *(r+i) = 0;
  cula_exception(culaDgesvd('A', 'A', k, k, w, k, sval, lsmat, k, rsmat, k), "culaDgesvd", "for w^0.5 (this w is not whitening matrix)");

  for(i=0; i<k; i++)
      *(smat+(k*i+i)) = sqrt(*(sval+i));
  // computing square root of k \times k W via svd and 2 matrix mutiplications; reuse r for pinv(r)
  cula_exception(culaBlockDgemm('n', 'n', k, k, k, 1, lsmat, k, smat, k, 0, w, k), "culaBlockDgemm", "first multiplication for w");
  cula_exception(culaBlockDgemm('n', 'n', k, k, k, 1, r, k, rsmat, k, 0, w, k), "culaBlockDgemm", "second multiplication for w");

  // since matrix multiplication is associative, the following order is more efficient than the fifo order
  cula_exception(culaBlockDgemm('t', 't', k, k, k, 1, r, k, w, k, 0, r, k), "culaBlockDgemm", "pinv(R)'*(W^0.5)'");
  cula_exception(culaBlockDgemm('n', 'n', n, k, k, 1, a, n, r, k, 0, white, n), "culaBlockDgemm", "Q*(pinv(R)'*W^0.5)'"); // can be blocked; this computes the whitening matrix into the location pointed to by the pointer white

  free(reflectors);
  free(s);
  free(c);
  free(r);
  free(w);
  free(lsmat);
  free(smat);
  free(rsmat);
  free(sval);

  return culaNoError;
}


culaStatus pinv_nys_asym(int m, int n, int k, double *a, double *b) // pseudoinverse of an asymmetric matrix using nystrom random projection method: a is the input matrix (m \times n), b is the output matrix (n \times m); k is the reduced dimensionality used for qr, etc.
{
  int i; // loop counter for setting the digaonal of the k \times k pinv_sig matrix to inverse square root elementwise; done in cpu because k is small
  // computing the row span, i.e., a*a'
  double *aat = malloc(sizeof(double)*m*m);
  if(aat == NULL)
  {
    printf("malloc failed at aat\n");
    return culaDataError;
  }
  memset(aat, 0, sizeof(double)*m*m);
  // computing the column span, i.e., a'*a
  double *ata = malloc(sizeof(double)*n*n);
  if(ata == NULL)
  {
    printf("malloc failed at ata\n");
    return culaDataError;
  }
  memset(ata, 0, sizeof(double)*n*n);

  double *uaat = malloc(sizeof(double)*m*k);
  if(uaat == NULL)
  {
    printf("malloc failed at uaat\n"); fflush(stdout);
    return culaDataError;
  }
  memset(uaat, 0, sizeof(double)*m*k);
  double *uata = malloc(sizeof(double)*n*k);
  if(uata == NULL)
  {
    printf("malloc failed at uata\n"); fflush(stdout);
    return culaDataError;
  }
  memset(uata, 0, sizeof(double)*n*k);
  double *l = malloc(sizeof(double)*k*n);
  if(l == NULL)
  {
    printf("malloc failed at l\n"); fflush(stdout);
    return culaDataError;
  }
  memset(l, 0, sizeof(double)*k*n);
  double *pinv_sig = malloc(sizeof(double)*k*k);
  if(pinv_sig == NULL)
  {
    printf("malloc failed at pinv_sig\n"); fflush(stdout);
    return culaDataError;
  }
  memset(pinv_sig, 0, sizeof(double)*k*k);

  double *reflectors = malloc(sizeof(double)*k);
  if(reflectors == NULL)
  {
    printf("malloc failed at reflectors\n"); fflush(stdout);
    return culaDataError;
  }
  memset(reflectors, 0, sizeof(double)*k);

  printf("computing A A^transpose\n"); // m \times m
  cula_exception(culaBlockDgemm('n', 't', m, m, n, 1, a, m, a, m, 0, aat, m), "culaBlockDgemm", "A A^transpose");
  cula_exception(nystrom_whitening(m, k, aat, uaat), "nystrom whitening within asymmetric pseudoinverse (c malloc must have failed)", "U");
  cula_exception(culaDgeqrfp(m, k, uaat, m, reflectors), "culaDgeqrfp", "qr step 1 in pinv using nys");
  cula_exception(culaDorgqr(m, k, k, uaat, m, reflectors), "culaDorgqr", "qr step 2 in pinv using nys"); // Generate Q and reuse u for n \times k matrix

  // note: no need to memset reflectors

  printf("computing A^transpose A\n"); // n \times n
  cula_exception(culaBlockDgemm('t', 'n', n, n, m, 1, a, m, a, m, 0, ata, n), "culaBlockDgemm", "A^transpose A");
  cula_exception(nystrom_whitening(n, k, ata, uata), "nystrom whitening within asymmetric pseudoinverse (c malloc must have failed)", "U");
  cula_exception(culaDgeqrfp(n, k, uata, n, reflectors), "culaDgeqrfp", "qr step 1 in pinv using nys");
  cula_exception(culaDorgqr(n, k, k, uata, n, reflectors), "culaDorgqr", "qr step 2 in pinv using nys"); // Generate Q and reuse u for n \times k matrix

  // compute just one of the diag matrices for faster computation (sigma2 done below); l is actually k \times k but also using it an an intermediate buffer, so malloced above k \times n
  cula_exception(culaBlockDgemm('t', 'n', k, n, n, 1, uata, n, ata, n, 0, l, k), "culaBlockDgemm", "u'*b");
  cula_exception(culaBlockDgemm('n', 'n', k, k, n, 1, l, k, uata, n, 0, l, k), "culaBlockDgemm", "u'*b  *u");

  memset(pinv_sig, 0, sizeof(double)*k*k);
  for(i=0; i<k; i++) // loop for elementwise square root of the small k \times k diagonal matrix
  {
    if( fabs(*( l+((k+1)*i) )) > PINV_TOL )
      *(pinv_sig+( (k+1)*i )) = 1/sqrt( *( l+((k+1)*i) ) );
  }

  // final two multiplicationsfor the pseudoinverse (using the svd computed above using nystrom)
  cula_exception(culaBlockDgemm('n', 'n', n, k, k, 1, uata, n, pinv_sig, k, 0, b, n), "culaBlockDgemm", "V*pinv_sig");
  cula_exception(culaBlockDgemm('n', 't', n, m, k, 1, b, n, uaat, m, 0, b, n), "culaBlockDgemm", "V*pinv_sig *U'");
  printf("nystrom pseudoinverse done\n");

  free(aat);
  free(ata);
  free(uaat);
  free(uata);
  free(l);
  free(pinv_sig);
  free(reflectors);

  return culaNoError;
}
