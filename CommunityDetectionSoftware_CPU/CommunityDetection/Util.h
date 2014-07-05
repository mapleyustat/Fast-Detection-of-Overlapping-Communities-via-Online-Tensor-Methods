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
#ifndef __CommunityDetection__Util__
#define __CommunityDetection__Util__
#include "stdafx.h"
int furong_atoi(string word);
double furong_atof(string word);

void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, \
	SparseMatrix<double> &W, SparseMatrix<double> &Z_B_part1, SparseMatrix<double> &Z_B_part2, SparseMatrix<double> &Z_C_part1, SparseMatrix<double> &Z_C_part2, VectorXd &mu_a, VectorXd &mu_b, VectorXd &mu_c);
void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, \
	SparseMatrix<double> D_b_mat, VectorXd D_b_mu, \
	SparseMatrix<double> D_c_mat, VectorXd D_c_mu, \
	VectorXd &lambda, MatrixXd & phi_new);
VectorXd tensor_form_main(SparseMatrix<double> D_a_mat, \
	SparseMatrix<double> D_b_mat, \
	SparseMatrix<double> D_c_mat, \
	VectorXd curr_eigenvec);
VectorXd tensor_form_shift1(MatrixXd Pair_ab, MatrixXd Pair_ac, MatrixXd Pair_bc, \
	VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec);
VectorXd tensor_form_shift0(VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec);
void tensorDecom_alpha0_online(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, \
	SparseMatrix<double> D_b_mat, VectorXd D_b_mu, \
	SparseMatrix<double> D_c_mat, VectorXd D_c_mu, \
	VectorXd &lambda, MatrixXd & phi_new);
MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, \
	VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, \
	MatrixXd phi, double learningrate);
VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, \
	VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi);


Eigen::VectorXd concatenation_vector(Eigen::VectorXd A, Eigen::VectorXd B);
#endif