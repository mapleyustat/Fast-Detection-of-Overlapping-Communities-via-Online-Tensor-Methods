/*
  This code for the mixed membership community project was written by Furong Huang and
  are copyrighted under the (lesser) GPL:
  Copyright (C) 2013 Furong Huang
  This program is free software; you can redistribute it and/or modify it under the terms of the
  GNU Lesser General Public License as published by the Free Software Foundation;
  version 3.0 or later. This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE.
  See the GNU Lesser General Public License for more details. You should have received a copy of
  the GNU Lesser General Public License along with this program;
  if not, write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
  02111-1307, USA.
  The authors may be contacted via email at: furongh(at)uci(.)edu
*/

#ifndef __Community__Community__
#define __Community__Community__

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseQR"
#include "SVDLIBC/svdlib.h" // change the path as needed
#include <set>
#include <queue>
#include <string>
#include <map>
#include <math.h>
#include <iterator>     // std::advance
#include <sys/time.h>


using namespace std;
using namespace Eigen;

// number of nodes in the partitions
#define NX 263517// 65879//263517//10010// 263517
#define NA 263517// 65879//263516//9529//263516
#define NB 263517//65879//263517//9530//263517
#define NC 263517//65880//263516//9529//263516
#define KHID 10
#define KTRUE 6003//159//6003
//#define ErrCal
#define NUM_THRESH 8
#define thresh_vec_def {0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1,0.08}

#define alpha0  1.0 // dirichlet concentration parameter
#define LEARNRATE 1e-9 // learning rate for tensor decomposition
#define MINITER 10000 // minimum number of iterations
#define MAXITER 100000 // maximum number of iterations

#define TOLERANCE 1e-4
#define EPS 1e-6
#define BINARY // binary case
//#define EXPECTED // weighted case
#define pinvtoler 1e-6
#define PVALUE_TOLE 0.01 // pvalue tolerance


// dataset paths
#define PATH "./data/"
#define FILE_PI_WRITE PATH "PI_WRITE.txt"
#define FILE_WHITE_WRITE PATH "WHITE_WRITE.txt"
#define FILE_ZB_WRITE PATH "ZB_WRITE.txt"
#define FILE_ZC_WRITE PATH "ZC_WRITE.txt"
#define FILE_INVLAMPHI_WRITE PATH "INVLAMPHI_WRITE.txt"

#define FILE_GA PATH "Gx_a.txt"
#define FILE_GB PATH "Gx_b.txt"
#define FILE_GC PATH "Gx_c.txt"
#define FILE_Gb_a PATH "Gb_a.txt"
#define FILE_Gb_c PATH "Gb_c.txt"
#define FILE_Gc_a PATH "Gc_a.txt"

#define FILE_Pi_a PATH "Pi_true_a.txt"
#define FILE_Pi_b PATH "Pi_true_b.txt"
#define FILE_Pi_c PATH "Pi_true_c.txt"
#define FILE_Pi_x PATH "Pi_true_x.txt"


#define edgeD_MAX -log(0.94) // this is only correct when distances are normalized to be smaller than



int write_pi(char *filename, SparseMatrix<double> spmat);

//Furong's function for calculating p-values 
double CalculateMean(double value[], long len);
double CalculateVariance(double value[], long len);
double CalculateSampleVariance(double value[], long len);
double Calculate_StandardDeviation(double value[], long len);
double Calculate_SampleStandardDeviation(double value[], long len);
double Calculate_Covariance(double x[], double y[], long len);
double Calculate_Correlation(double x[], double y[], long len);
double Calculate_Tstat(double x[], double y[], long len);
double betacf(double a, double b, double x);
double gammln(double xx);
double betainc(double a, double b, double x);
double Calculate_Pvalue(double x[], double y[], long len);
void furongprintVector(double value[], long len, char *character);

// set of svd functions
std::pair<Eigen::MatrixXd, Eigen::VectorXd> k_svd (Eigen::MatrixXd A, int k);
std::pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> k_svd_observabe (Eigen::MatrixXd A, int k);
std::pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > k_svd_observabe (Eigen::SparseMatrix<double> A, int k);
pair<pair<Eigen::MatrixXd,Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd (Eigen::MatrixXd A);
//

double pinv_num (double pnum);
Eigen::VectorXd pinv_vector (Eigen::VectorXd pinvvec);
Eigen::SparseVector<double> pinv_vector (Eigen::SparseVector<double> pinvvec);
Eigen::MatrixXd pinv_matrix( Eigen::MatrixXd pinvmat);
Eigen::SparseMatrix<double> pinv_matrix( Eigen::SparseMatrix<double> pinvmat);
Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat);
Eigen::SparseMatrix<double> sqrt_matrix(Eigen::SparseMatrix<double> pinvmat);

// Nystrom sparse svd for sym and asym
SparseMatrix<double> random_embedding_mat(long cols_num, int k);
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> A,int k);
pair<SparseMatrix<double>,SparseMatrix<double> >  pinv_symNystrom_sparse(SparseMatrix<double> X);
SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X);
//
pair<pair<SparseMatrix<double>,SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X,int k);
pair<SparseMatrix<double>,SparseMatrix<double> > pinv_asymNystrom_sparse(SparseMatrix<double> X);
SparseMatrix<double> sqrt_asymNystrom_sparse(SparseMatrix<double> X);
//////////////////////////////////

void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C, VectorXd &mu_a, VectorXd &mu_b, VectorXd &mu_c);

void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new);
MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g,VectorXd Data_c_g,VectorXd Data_a_mu,VectorXd Data_b_mu,VectorXd Data_c_mu, Eigen::MatrixXd phi_old,double beta);
VectorXd The_second_term(VectorXd Data_a_g,VectorXd Data_b_g,VectorXd Data_c_g,VectorXd Data_a_mu,VectorXd Data_b_mu,VectorXd Data_c_mu,VectorXd phi);


//////////////////////////

Eigen::MatrixXd concatenation_matrix (Eigen::MatrixXd A, Eigen::MatrixXd B);
Eigen::VectorXd concatenation_vector (Eigen::VectorXd A, Eigen::VectorXd B);

typedef struct
{
    Eigen::VectorXd singular_values;
    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
}EigenSparseSVD; // struct interface between svdlibc and eigen

EigenSparseSVD sparse_svd(Eigen::SparseMatrix<double> eigen_sparse_matrix, int rank=0);

Eigen::MatrixXd normc(Eigen::MatrixXd phi);
Eigen::VectorXd normProbVector(Eigen::VectorXd P);
Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P);
Eigen::MatrixXd Condi2Joint(Eigen::MatrixXd Condi, Eigen::VectorXd Pa);
Eigen::MatrixXd joint2conditional(Eigen::MatrixXd edgePot,Eigen::VectorXd pa);// pa is the second dimension
Eigen::MatrixXd joint2conditional(Eigen::MatrixXd edgePot,Eigen::VectorXd pa);
Eigen::MatrixXd condi2condi(Eigen::MatrixXd p_x_h, Eigen::VectorXd p_h);

////////////////////////////////////////////////////////////////This is for Matrix generation

Eigen::SparseMatrix<double> read_G_sparse(char *file_name, char *G_name,int N1, int N2);
#endif /* defined(__Community__Community__) */
