//
//  MathProbabilities.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2013 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#pragma once
#include "stdafx.h"

using namespace Eigen;
using namespace std;
Eigen::MatrixXd normc(Eigen::MatrixXd phi)
{
	for (int i = 0; i < phi.cols(); i++)
	{
		phi.col(i).normalize();
	}

	return phi;
}
Eigen::SparseMatrix<double> normc(Eigen::SparseMatrix<double> phi)
{
	MatrixXd phi_f = normc((MatrixXd)phi);

	return phi_f.sparseView();
}
////////////////////////////////////////////////////////////

Eigen::VectorXd normProbVector(VectorXd P_vec)
{
	VectorXd P_norm = P_vec;
	if (P_vec == Eigen::VectorXd::Zero(P_vec.size())){
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec(row_idx) > 0) ? (P_positive + P_vec(row_idx)) : P_positive;
			P_negative = (P_vec(row_idx) > 0) ? P_negative : (P_negative + P_vec(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm(row_idx) = (P_norm(row_idx)<0) ? 0 : P_norm(row_idx);
		}
	}
	return P_norm;
}

Eigen::SparseVector<double> normProbVector(Eigen::SparseVector<double> P_vec)
{
	VectorXd P_dense_vec = (VectorXd)P_vec;
	Eigen::SparseVector<double> P_norm;
	if (P_dense_vec == VectorXd::Zero(P_vec.size()))
	{
		P_norm = P_vec;
	}
	else{
		double P_positive = 0; double P_negative = 0;

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_positive = (P_vec.coeff(row_idx) > 0) ? (P_positive + P_vec.coeff(row_idx)) : P_positive;
			P_negative = (P_vec.coeff(row_idx) > 0) ? P_negative : (P_negative + P_vec.coeff(row_idx));
		}
		if (fabs(P_positive) < fabs(P_negative)){
			P_norm = -P_vec / fabs(P_negative);
		}
		else{
			P_norm = P_vec / fabs(P_positive);
		}

		for (int row_idx = 0; row_idx < P_vec.size(); row_idx++){
			P_norm.coeffRef(row_idx) = (P_norm.coeff(row_idx)<0) ? 0 : P_norm.coeff(row_idx);
		}
	}
	P_norm.prune(TOLERANCE);
	return P_norm;
}

Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P)
{
	// each column is a probability simplex
	Eigen::MatrixXd P_norm(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		Eigen::VectorXd P_vec = P.col(col);
		P_norm.col(col) = normProbVector(P_vec);
	}
	return P_norm;
}

Eigen::SparseMatrix<double> normProbMatrix(Eigen::SparseMatrix<double> P)
{
	// each column is a probability simplex
	Eigen::SparseMatrix<double> P_norm;
	P_norm.resize(P.rows(), P.cols());
	for (int col = 0; col < P.cols(); col++)
	{
		//SparseVector<double> A_col_sparse = A_sparse.block(0, i, A_sparse.rows(),1);
		SparseVector<double> P_vec = P.block(0, col, P.rows(), 1);
		SparseVector<double> P_vec_norm;
		P_vec_norm.resize(P_vec.size());
		P_vec_norm = normProbVector(P_vec);
		for (int id_row = 0; id_row < P.rows(); id_row++)
		{
			P_norm.coeffRef(id_row, col) = P_vec_norm.coeff(id_row);
		}
	}
	P_norm.makeCompressed();
	P_norm.prune(TOLERANCE);
	return P_norm;
}
