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
extern double alpha0;
extern int KHID;
extern int NX;
extern int NA;
extern int NB;
extern int NC;
extern const char * FILE_Gb_a;
extern const char * FILE_Gc_a;
using namespace Eigen;
using namespace std;

int furong_atoi(string word)
{
	int lol = atoi(word.c_str()); /*c_str is needed to convert string to const char*
								  previously (the function requires it)*/
	return lol;
}
double furong_atof(string word)
{
	double lol = atof(word.c_str()); /*c_str is needed to convert string to const char*
									 previously (the function requires it)*/
	return lol;
}

// set of whitening matrix
void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, \
	SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C, VectorXd &mu_a, VectorXd &mu_b, VectorXd &mu_c)
{
	double nx = (double)Gx_a.rows();
	double inv_nx;
#ifdef NORMALIZE
	inv_nx = 1 / nx;
#else
	inv_nx = 1;
#endif
	SparseVector<double> my_ones = (VectorXd::Ones((int)nx)).sparseView();
	SparseVector<double> mu_a_sparse = my_ones.transpose() * Gx_a;	SparseVector<double> mu_b_sparse = my_ones.transpose() * Gx_b;	SparseVector<double> mu_c_sparse = my_ones.transpose() * Gx_c;
	mu_a_sparse = inv_nx * mu_a_sparse;	mu_b_sparse = inv_nx * mu_b_sparse;	mu_c_sparse = inv_nx * mu_c_sparse;
	mu_a = (VectorXd)mu_a_sparse; mu_b = (VectorXd)mu_b_sparse; mu_c = (VectorXd)mu_c_sparse;

	//    cout << "--------------starting to calculate Z_B, Z_C implicitly---------------------------"<< endl;
	SparseMatrix<double> Z_B_numerator = Gx_a.transpose() * Gx_c;// NA * NC
	Z_B_numerator = inv_nx * Z_B_numerator;       // NA * NC
	SparseMatrix<double> Z_B_denominator = Gx_b.transpose()*Gx_c;// NB * NC
	Z_B_denominator = inv_nx * Z_B_denominator;//NB * NC

	SparseMatrix<double> Z_C_numerator = Gx_a.transpose() * Gx_b; // NA * NB
	Z_C_numerator = inv_nx *  Z_C_numerator;       // NA * NB
	SparseMatrix<double> Z_C_denominator = Gx_c.transpose()*Gx_b;//NC * NB
	Z_C_denominator = inv_nx * Z_C_denominator;//NC * NB

	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans_Zc = pinv_Nystrom_sparse_component(Z_C_denominator);
	//X_pinv = V_invL_Utrans.first.first * V_invL_Utrans.second * V_invL_Utrans.first.second;
	pair<pair<SparseMatrix<double>, SparseMatrix<double> >, SparseMatrix<double> > V_invL_Utrans_Zb = pinv_Nystrom_sparse_component(Z_B_denominator);

	//    cout << "--------------starting to calculate Z_B, Z_C ---------------------------"<< endl;
	// compute M2 Main Term
	double para_main = (alpha0 + 1);
	// NA by k term: 
	SparseMatrix<double> term_na_k = (Z_C_numerator * V_invL_Utrans_Zc.first.first) * V_invL_Utrans_Zc.second;
	Z_C = term_na_k * V_invL_Utrans_Zc.first.second;
	//	cout << "Z_C_numerator: " << endl << (MatrixXd)Z_C_numerator << endl;
	//	cout << "Z_C_denominator: " << endl << (MatrixXd)Z_C_denominator << endl;
	//	cout << "Z_C: " << endl << (MatrixXd)Z_C << endl;

	// k by k term: 
	SparseMatrix<double> term_k_k = (V_invL_Utrans_Zc.first.second * Z_C_denominator) * V_invL_Utrans_Zb.first.second.transpose();

	// k by NA term: 
	SparseMatrix<double> term_k_na = (V_invL_Utrans_Zb.second * V_invL_Utrans_Zb.first.first.transpose()) * Z_B_numerator.transpose();
	Z_B = term_k_na.transpose() * V_invL_Utrans_Zb.first.second;
	//	cout << "Z_B_numerator: " << endl << (MatrixXd)Z_B_numerator << endl;
	//	cout << "Z_B_denominator: " << endl << (MatrixXd)Z_B_denominator << endl;
	//	cout << "Z_B: " << endl << (MatrixXd)Z_B << endl;
	SparseMatrix<double> M2 = (term_na_k * term_k_k) * term_k_na; M2 = para_main * M2;
	//	cout << "M2: " << M2 << endl;
	// compute M2 Shift Term
	double para_shift = alpha0;
	//    cout <<"-------------------computing square_mu_a_sparse--------"<<endl;
	SparseMatrix<double> shiftTerm = mu_a_sparse * mu_a_sparse.transpose(); shiftTerm = para_shift * shiftTerm;
	SparseMatrix<double> M2_a = M2 - shiftTerm;	M2_a.makeCompressed(); M2_a.prune(TOLERANCE);
	//    cout << "-----------M2_alpha0:nonZeros()" << M2.nonZeros()<< "-------------"<<endl;

	pair< SparseMatrix<double>, SparseVector<double> > Vw_Lw = SVD_symNystrom_sparse(M2_a);
	SparseMatrix<double> Uw = Vw_Lw.first.leftCols(KHID);
	VectorXd Lw = (VectorXd)Vw_Lw.second;
	Lw = pinv_vector(Lw.head(KHID).cwiseSqrt());
	MatrixXd diag_Lw_sqrt_inv = Lw.asDiagonal();
	SparseMatrix<double> diag_Lw_sqrt_inv_s = diag_Lw_sqrt_inv.sparseView();
	W.resize(Gx_a.cols(), KHID);
	W = Uw * diag_Lw_sqrt_inv_s;
	W.makeCompressed(); W.prune(TOLERANCE);
	//	cout << "---------------------dimension of W : " << W.rows() << " , " << W.cols() << "----------------" << endl;
	//	cout << "-----------End of Whitening----------nonZeros() of W : " << W.nonZeros() << endl;

}


///////////////////
void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new)
{
	double error; MatrixXd phi_old;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	double inv_x = 1.0 / ((double)D_a_mat.cols());
	double para_main = (alpha0 + 1.0)*(alpha0 + 2.0) / 2.0 * inv_x;
	double para_shift1 = -alpha0 *(alpha0 + 1.0) / 2.0 * inv_x;
	double para_shift0 = alpha0*alpha0;

	phi_new = qr.householderQ();
	lambda = VectorXd::Zero(KHID);
	SparseMatrix<double> pair_ab = D_a_mat * D_b_mat.transpose(); MatrixXd Pair_ab = (MatrixXd)pair_ab;
	SparseMatrix<double> pair_ac = D_a_mat * D_c_mat.transpose(); MatrixXd Pair_ac = (MatrixXd)pair_ac;
	SparseMatrix<double> pair_bc = D_b_mat * D_c_mat.transpose(); MatrixXd Pair_bc = (MatrixXd)pair_bc;
	//	cout << "one sample a: " << (VectorXd) D_a_mat.block(0, 0, KHID, 1) << endl;
	//	cout << "one sample b: " << (VectorXd) D_b_mat.block(0, 0, KHID, 1) << endl;
	//	cout << "one sample c: " << (VectorXd) D_c_mat.block(0, 0, KHID, 1) << endl;
	A_random.resize(0, 0);
	long iteration = 1;
	//	cout << "phi_new: " << phi_new << endl;
	while (true)
	{
		long iii = iteration % NX;
		phi_old = phi_new;
		for (int index_k = 0; index_k < KHID; index_k++){
			VectorXd curr_eigenvec = phi_old.col(index_k);
			phi_new.col(index_k) = para_main * tensor_form_main(D_a_mat, D_b_mat, D_c_mat, curr_eigenvec) \
				+ para_shift1 * tensor_form_shift1(Pair_ab, Pair_ac, Pair_bc, D_a_mu, D_b_mu, D_c_mu, curr_eigenvec)\
				+ para_shift0*tensor_form_shift0(D_a_mu, D_b_mu, D_c_mu, curr_eigenvec);

		}
		//		cout << "phi_new: " << phi_new << endl;
		lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
		phi_new = normc(phi_new);
		//		cout << "phi_new: " << phi_new << endl;
		//		cout << "lambda: " << lambda << endl;
		if (iteration < MINITER){}
		else
		{
			error = (phi_new - phi_old).norm();
			cout << "error: " << error << endl;
			if (error < TOLERANCE || iteration > MAXITER)
			{
				cout << " coverged iteration: " << iteration << endl;
				break;
			}
		}

		iteration++;
	}
}


VectorXd tensor_form_main(SparseMatrix<double> D_a_mat, SparseMatrix<double> D_b_mat, SparseMatrix<double> D_c_mat, VectorXd curr_eigenvec){
	VectorXd D_b_f = D_b_mat.transpose() * curr_eigenvec;
	VectorXd D_c_f = D_c_mat.transpose() * curr_eigenvec;
	VectorXd D_bc_f = D_b_f.cwiseProduct(D_c_f);
	return D_a_mat*D_bc_f;
}
VectorXd tensor_form_shift1(MatrixXd Pair_ab, MatrixXd Pair_ac, MatrixXd Pair_bc, VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec){
	double coeff_uA = curr_eigenvec.transpose() * Pair_bc * curr_eigenvec;
	double coeff_uB = curr_eigenvec.transpose() * D_b_mu;
	double coeff_uC = curr_eigenvec.transpose() * D_c_mu;
	VectorXd coeff_A = coeff_uA * D_a_mu;
	VectorXd coeff_B = coeff_uB * (Pair_ac * curr_eigenvec);
	VectorXd coeff_C = coeff_uC * (Pair_ab * curr_eigenvec);
	return coeff_A + coeff_B + coeff_C;
}
VectorXd tensor_form_shift0(VectorXd D_a_mu, VectorXd D_b_mu, VectorXd D_c_mu, VectorXd curr_eigenvec){
	double coeff1 = curr_eigenvec.transpose()*D_b_mu;
	double coeff2 = curr_eigenvec.transpose()*D_c_mu;
	return (coeff1*coeff2)* D_a_mu;
}


void tensorDecom_alpha0_online(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new)
{
	double error;
	MatrixXd A_random(MatrixXd::Random(KHID, KHID));
	MatrixXd phi_old;

	A_random.setRandom();
	HouseholderQR<MatrixXd> qr(A_random);
	phi_new = qr.householderQ();
	lambda = VectorXd::Zero(KHID);
	A_random.resize(0, 0);
	long iteration = 1;
	cout << "phi_new: " << phi_new << endl;
	while (true)
	{
		long iii = iteration % NX;
		VectorXd D_a_g = D_a_mat.col((int)iii);//
		VectorXd D_b_g = D_b_mat.col((int)iii);
		VectorXd D_c_g = D_c_mat.col((int)iii);
		double learningrate = min(1e-9, 1.0 / sqrt((double)iteration));

		phi_old = phi_new;
		phi_new = Diff_Loss(D_a_g, D_b_g, D_c_g, D_a_mu, D_b_mu, D_c_mu, phi_old, learningrate);
		cout << "phi_new: " << phi_new << endl;
		///////////////////////////////////////////////
		if (iteration < MINITER)
		{
		}
		else
		{
			error = (normc(phi_new) - normc(phi_old)).norm();
			cout << "error: " << error << endl;
			if (error < TOLERANCE || iteration > MAXITER)
			{

				break;

			}
		}

		iteration++;
	}
	lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0 / 2.0)).transpose();
	phi_new = normc(phi_new);
	cout << "lambda: " << lambda << endl;
	cout << "phi_new: " << endl << phi_new << endl;
}

MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, MatrixXd phi, double learningrate)
{
	MatrixXd New_Phi;
	double theta = 10000;

	MatrixXd myvectors = MatrixXd::Zero(KHID, KHID);
	cout << "phi: " << endl << phi << endl;
	cout << "learningrate: " << learningrate << endl;
	cout << "Data_a_g: " << Data_a_g.transpose() << endl;
	cout << "Data_b_g: " << Data_b_g.transpose() << endl;
	cout << "Data_c_g: " << Data_c_g.transpose() << endl;
	for (int index_k = 0; index_k < KHID; index_k++)
	{
		VectorXd curr_eigenvec = phi.col(index_k);
		VectorXd SquareTerm = (curr_eigenvec.transpose()*phi).array().pow(2).transpose();
		MatrixXd The_first_term_noSum = phi * SquareTerm.asDiagonal();
		VectorXd vector_term1 = (3.0*theta) * The_first_term_noSum.rowwise().sum();
		cout << "vector_term1: " << vector_term1.transpose() << endl;
		VectorXd vector_term2 = -3.0 * The_second_term(Data_a_g, Data_b_g, Data_c_g, Data_a_mu, Data_b_mu, Data_c_mu, curr_eigenvec);
		cout << "vector_term2: " << vector_term2.transpose() << endl;
		myvectors.col(index_k) = vector_term1 + vector_term2;

	}
	cout << "myvectors: " << endl << myvectors << endl;
	New_Phi = phi - myvectors*learningrate;
	cout << "New_Phi: " << endl << New_Phi << endl;
	return New_Phi;
}

VectorXd The_second_term(VectorXd Data_a_g, VectorXd Data_b_g, VectorXd Data_c_g, VectorXd Data_a_mu, VectorXd Data_b_mu, VectorXd Data_c_mu, VectorXd phi)
{
	// phi is a VectorXd
	double para0 = (alpha0 + 1.0)*(alpha0 + 2.0) / 2.0;
	double para1 = alpha0*alpha0;
	double para2 = -alpha0 *(alpha0 + 1.0) / 2.0;

	VectorXd Term1 = para0*(phi.dot(Data_a_g))*(phi.dot(Data_b_g)) * Data_c_g;
	VectorXd Term2 = para1*(phi.dot(Data_a_mu))*(phi.dot(Data_b_mu))*Data_c_mu;
	VectorXd Term31 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_g))*Data_c_mu;
	VectorXd Term32 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_mu))*Data_c_g;
	VectorXd Term33 = para2*(phi.dot(Data_a_mu))*(phi.dot(Data_b_g))*Data_c_g;
	VectorXd output = Term1 + Term2 + Term31 + Term32 + Term33;
	cout << "(phi.dot(Data_a_g)): " << (phi.dot(Data_a_g)) << endl;
	cout << "(phi.dot(Data_b_g)): " << (phi.dot(Data_b_g)) << endl;
	cout << "Data_c_g: " << Data_c_g.transpose() << endl;
	cout << "output: " << output.transpose() << endl;
	return output;
}



//void Unwhitening(VectorXd lambda, MatrixXd eignevec, SparseMatrix<double> whitening, \
//	SparseMatrix<double> sym_x1, SparseMatrix<double> sym_x2, \
//	VectorXd & p_h, SparseMatrix<double> &p_x1_h, SparseMatrix<double> &p_x2_h, SparseMatrix<double> &p_x3_h)
//// sym_x1 is Z_B; sym_x2 is Z_B;
//{
//    VectorXd p_h = normProbVector(lambda.array().pow(-2));	p_h = p_h / p_h.sum();
//
//	Eigen::SparseMatrix<double> p_x1_h; p_x1_h.resize(NA, KHID);
//	Eigen::SparseMatrix<double> p_x2_h; p_x2_h.resize(NB, KHID);
//	Eigen::SparseMatrix<double> p_x3_h; p_x3_h.resize(NC, KHID);
//
//
//	if (whitening.nonZeros() < KHID + 1)
//	{
//	MatrixXd tmp_p_x1_h = MatrixXd::Ones(NX, KHID);
//	SparseMatrix<double> tmp_p_x1_h_sparse = tmp_p_x1_h.sparseView();
//	p_x1_h = normProbMatrix(tmp_p_x1_h_sparse);
//	MatrixXd tmp_p_x2_h = MatrixXd::Ones(NX, KHID);
//	SparseMatrix<double> tmp_p_x2_h_sparse = tmp_p_x2_h.sparseView();
//	p_x2_h = normProbMatrix(tmp_p_x2_h_sparse);
//	MatrixXd tmp_p_x3_h = MatrixXd::Ones(NX, KHID);
//	SparseMatrix<double> tmp_p_x3_h_sparse = tmp_p_x3_h.sparseView();
//	p_x3_h = normProbMatrix(tmp_p_x3_h_sparse);
//
//	}
//	else
//	{
//	Eigen::MatrixXd Lambda_diag = lambda.asDiagonal();
//	Eigen::SparseMatrix<double> Lambda_diag_sparse;
//	Lambda_diag_sparse.resize(KHID, KHID);
//	Lambda_diag_sparse = Lambda_diag.sparseView();
//	Lambda_diag_sparse.makeCompressed();
//	Lambda_diag_sparse.prune(TOLERANCE);
//
//	Eigen::SparseMatrix<double> eigenvec_sparse;
//	eigenvec_sparse.resize(KHID, KHID);
//	eigenvec_sparse = eignevec.sparseView();
//	SparseMatrix<double> pair_pinv_tmp1 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()*sym_x1));// this should be k * VOCA_SIZE
//	SparseMatrix<double> pair_pinv_tmp2 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()*sym_x2));
//	SparseMatrix<double> pair_pinv_tmp3 = pinv_aNystrom_sparse((SparseMatrix<double>)(whitening.transpose()));
//	p_x1_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp1*eigenvec_sparse * Lambda_diag_sparse));
//	p_x2_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp2*eigenvec_sparse * Lambda_diag_sparse));
//	p_x3_h = normProbMatrix((SparseMatrix<double>)(pair_pinv_tmp3*eigenvec_sparse * Lambda_diag_sparse));
//	}
//
//}

///////////////////////////////// old files... 
Eigen::VectorXd concatenation_vector(Eigen::VectorXd A, Eigen::VectorXd B) // concatenate 2 eigen vectors
{
	Eigen::VectorXd C(A.size() + B.size());
	// C.resize(A.size()+B.size());
	C << A, B;
	return C;
}

