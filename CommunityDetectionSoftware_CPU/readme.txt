/////////Author: Furong Huang: Webpage: https://sites.google.com/site/furongfionahuang/
/////////University of California, Irvine

For all the notations, refer to the paper "Fast Detection of Overlapping Communities via Online Tensor Methods", arxiv preprint:
http://arxiv.org/abs/1309.0787

General instructions:
1. Eigen and SVDLIBC packages are required for running our Community program. Download both into the current directory; else, change the paths in Community.h and the makefile.
2. Firstly cd to SVDLIBC to make.
3. Secondly cd to root to make. 
4. You might have to change file path accordingly in Community.h.
5. There is an option for calculating error, you can switch that on by uncommeningt #define ErrCal, or switch it off by commenting it off. When you switch that on, you have to specify your pi_est threshold accordingly, which can be changed in your main.cpp. Of course, NUM_THRE(defined in Community.h) should agree with the number of thresholds you are using.

Input:
Adjacency matrices in the sparse format (row, column, value). Set the flags in Community.h accordingly.

Output:
Error and match values are output on the screen.

Code structure:
main.cpp is the main file. For the various functions called, refer accordingly to the Community.cpp file. For the input and output of each function, refer to the argument names (they are similar to the names in the paper).
The two important functions are:
1. second_whiten(Gx_a,Gx_b,Gx_c,W,Z_B,Z_C,mu_a,mu_b,mu_c);
// This is used for whitening. The adjacency submatrices are prefixed with G and mean vectors with mu. The first 3 arguments and the last 3 arguments are the inputs. The middle 3 arguments are the outputs where W = whitening matrix, Z_B and Z_C are the transition matrices as in the paper.
2. tensorDecom_alpha0(Data_a_G,Data_a_mu,Data_b_G,Data_b_mu,Data_c_G,Data_c_mu,lambda,phi_new);
// This is used for implicit tensor decomposition. The whitened data vectors are prefixed with Data. The eigenvalues are in lambda and eigenvectors are in phi_new. The first 6 arguments are the input (mode vectors of a symmetric orthogonal tensor). The last 2 arguments are the outputs.
