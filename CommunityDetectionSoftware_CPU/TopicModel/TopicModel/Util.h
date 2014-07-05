//
//  Util.h
//  latenttree
/*******************************************************
* Copyright (C) 2013 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {Community Detection project}.
*
* All rights reserved.
*******************************************************/
#ifndef __TopicModel__Util__
#define __TopicModel__Util__
#include "stdafx.h"
int furong_atoi(string word);
double furong_atof(string word);

void second_whiten_topic(SparseMatrix<double> Gx_a, \
	SparseMatrix<double> &W, VectorXd &mu_a, SparseMatrix<double> &Uw, SparseMatrix<double> &diag_Lw_sqrt_inv_s, VectorXd &Lengths);
void tensorDecom_alpha0_topic(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, VectorXd Lengths, VectorXd &lambda, MatrixXd & phi_new);

VectorXd tensor_form_main_topic(SparseMatrix<double> Gx_aNorm0, SparseMatrix<double> Gx_aNorm1, \
	SparseMatrix<double> Gx_aNorm2, SparseMatrix<double> Gx_aSquareNorm, SparseMatrix<double> Gx_aCubNorm, VectorXd curr_eigenvec);


VectorXd tensor_form_shift1_topic(MatrixXd Pair_ab, VectorXd D_a_mu,  VectorXd curr_eigenvec);
VectorXd tensor_form_shift0_topic(VectorXd D_a_mu, VectorXd curr_eigenvec);
void tensorDecom_alpha0_online(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, \
	SparseMatrix<double> D_b_mat, VectorXd D_b_mu, \
	SparseMatrix<double> D_c_mat, VectorXd D_c_mu, \
	VectorXd &lambda, MatrixXd & phi_new);
MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, \
	VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, \
	MatrixXd phi, double learningrate);
VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, \
	VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi);
void Unwhitening(VectorXd lambda, MatrixXd eigenvec, SparseMatrix<double> Uw, SparseMatrix<double> diag_Lw_sqrt_inv, \
	VectorXd & alpha, MatrixXd &R_A);
void normProbVectorJohn(VectorXd V, VectorXd & P_norm);
int decode(VectorXd alpha, MatrixXd beta, VectorXd DOCLEN, SparseMatrix<double> corpus, char* filename);
void lda_inference(MatrixXd R, vector<int> my_doc_word, vector<double> my_doc_count, double total, VectorXd alpha, VectorXd& h, double& likelihood, int Maxiter);
double compute_likelihood(MatrixXd R, vector<int> my_doc_word, vector<double> my_doc_count, VectorXd alpha, double** phi, double* var_gamma);

VectorXd eigen_gamma(VectorXd a);
VectorXd vec_log(VectorXd a);
double log_sum(double log_a, double log_b);
double digamma(double x);
double log_gamma(double x);
void estimate_h_ll(SparseMatrix<double> R, vector<long> my_doc_word, vector<double> my_doc_count, VectorXd alpha, VectorXd& h, double& ll, int Maxiter);
Eigen::VectorXd concatenation_vector(Eigen::VectorXd A, Eigen::VectorXd B);
#endif