# This code for the mixed membership community project was written by Mohammad Umar Hakeem and Niranjan U N and
# are copyrighted under the (lesser) GPL:
# Copyright (C) 2013 Mohammad Umar Hakeem and Niranjan U N
# This program is free software; you can redistribute it and/or modify it under the terms of the
# GNU Lesser General Public License as published by the Free Software Foundation;
# version 3.0 or later. This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details. You should have received a copy of
# the GNU Lesser General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.
# The authors may be contacted via email at: mhakeem(at)uci(.)edu , un(.)niranjan(at)uci(.)edu


CXX=nvcc
CC=gcc
CFLAGS=-I$(CULA_INC_PATH) -I/usr/local/cuda-5.0/include/ -m64 -mcmodel=large -funroll-all-loops -O3
MKLFLAGS=-L/opt/intel/mkl/bin/ -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LDIR=-L$(CULA_LIB_PATH_64) -L/usr/local/cuda/lib64/
LIBS=-lcula_lapack -lcublas -lcudart -lpthread -lcula_core
CURAND_LIBS=-I/usr/local/cuda-5.0/include/ -L/usr/local/cuda/lib64 -lcurand

algo: stpm.h pre_proc.c gemm_pinv_nys.c stpm.c wrappers.cu curand_rand_mat.cu post_proc.c
	$(CXX) -c wrappers.cu -arch sm_20
	$(CXX) -c curand_rand_mat.cu -arch sm_20 -lcurand
	$(CC) -o algo stpm.c pre_proc.c post_proc.c wrappers.o curand_rand_mat.o gemm_pinv_nys.c $(LDIR) $(LIBS) $(CFLAGS) $(CURAND_LIBS)
	rm *.o
