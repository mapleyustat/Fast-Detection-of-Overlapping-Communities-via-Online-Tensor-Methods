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


#include <cuda.h>

#define MAX_BLOCKS 1024
#define MAX_THREADS 1024

__global__ void vecSq_kernel(double * dev_ptr, int len)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
		dev_ptr[tID] = pow(dev_ptr[tID],2);

	return; 
}

__global__ void vecRecprocalSqrt_kernel(double * dev_ptr, int len)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
		dev_ptr[tID] = rsqrt(dev_ptr[tID]);

	return; 
}

__global__ void vecSqrt_kernel(double * dev_ptr, int len)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
		dev_ptr[tID] = sqrt(dev_ptr[tID]);

	return; 
}

__global__ void vecInv_kernel(double * dev_ptr, int len, double tol)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
	{
		if(fabs(dev_ptr[tID]) < tol)
		{
			dev_ptr[tID] = 0;
		}
		else
		{
			dev_ptr[tID] = 1/dev_ptr[tID];
		}
	}
	return; 
}

__global__ void saveFrobNormVal_kernel(double * devBuf_ptr, double val)
{
	*devBuf_ptr = val;
	return; 
}


__global__ void genInvEvalMat_kernel(double * singVal_vec, double * sigma_mat, int m, int n, double *norm, double thrsh)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < (m*n))
	{
		if(((int)(tID % (m+1)) == 0)&&((int)(tID/(m+1)) < (int)(fmin((double)m,(double)n))))
		{
			if(fabs(singVal_vec[(int)(tID/(m+1))]) > thrsh)
			{ 
				sigma_mat[tID] = (*norm)/(singVal_vec[(int)(tID/(m+1))]);
			}
			else
			{
				sigma_mat[tID] = 0.0;
			}
		} 
		else
		{
			sigma_mat[tID] = 0.0;
		}
	}
	return; 
}


__global__ void fill_iter_mat_Vals_kernel(double *iter_mat_interm_mem_dev, double *iter_mat_location_ptr_dev, int len)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
	{
		iter_mat_location_ptr_dev[tID * 4] = iter_mat_interm_mem_dev[tID];
	}
	return; 
}


__global__ void pow3By2Evals_kernel(double *devBuf_ptr, int len)
{
	int tID = threadIdx.x + (blockIdx.x * blockDim.x);
	if(tID < len)
	{
		devBuf_ptr[tID] = pow(devBuf_ptr[tID], 1.5);
	}
	return; 
}


__global__ void l2Norm_kernel(double *devBuf_ptr, int len, double* out)
{
	int indx = 0;
	double accum = 0;
	
	for(indx = 0; indx < len; indx++)
	{
		accum += (devBuf_ptr[indx] * devBuf_ptr[indx]);
	}
	*out = sqrt(accum);
	return; 
}

__global__ void fillVal_kernel(double * dev_ptr)
{
		//dev_ptr[0] = -0.2e-7;
		dev_ptr[0] = 0.2;
		dev_ptr[1] = 0.7;
//		dev_ptr[2] = 0.5e-6;
		dev_ptr[2] = 5.6;
		dev_ptr[3] = -20.4;
		dev_ptr[4] = 4.4;						

	return; 
}


extern "C" void vecSq_CudaKer(double * dev_ptr, int len)
{
	vecSq_kernel<<<(len + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(dev_ptr, len);
}

extern "C" void vecInv_CudaKer(double * dev_ptr, int len, double tol)
{
	vecInv_kernel<<<(len + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(dev_ptr, len, tol);
}

extern "C" void vecSqrt_CudaKer(double *dev_ptr, int len)
{
	vecSqrt_kernel<<<(len + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(dev_ptr, len);
}


extern "C" void saveFrobNormVal_CudaKer(double *devBuf_ptr, double val)
{
	saveFrobNormVal_kernel<<<1,1>>>(devBuf_ptr, val);
}

extern "C" void l2Norm_CudaKer(double * devBuf_ptr, int len, double *out)
{
	l2Norm_kernel<<<1,1>>>(devBuf_ptr, len, out);
}

extern "C" void pow3By2Evals_CudaKer(double *devBuf_ptr, int len)
{
	pow3By2Evals_kernel<<<(len + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(devBuf_ptr, len);
}

extern "C" void vecRecprocalSqrt_CudaKer(double * dev_ptr, int len)
{
	vecRecprocalSqrt_kernel<<<(len + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(dev_ptr, len);
}

extern "C" void genInvEvalMat_CudaKer(double * singVal_vec, double * sigma_mat, int m, int n, double *norm, double thrsh)
{
	genInvEvalMat_kernel<<<((m*n) + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(singVal_vec, sigma_mat, m ,n, norm , thrsh);
}

extern "C" void fill_iter_mat_Vals_CudaKer(double *iter_mat_interm_mem_dev, double *iter_mat_location_ptr_dev, int len)
{
	fill_iter_mat_Vals_kernel<<<((len) + MAX_BLOCKS -1)/MAX_BLOCKS ,MAX_BLOCKS>>>(iter_mat_interm_mem_dev, iter_mat_location_ptr_dev, len);
}


extern "C" void fillVal_CudaKer(double * devBuf_ptr)
{
	fillVal_kernel<<<1,1>>>(devBuf_ptr);
}
