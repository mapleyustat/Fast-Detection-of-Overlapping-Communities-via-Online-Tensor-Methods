//
//  main
//  TopicModel.cpp
//
//  Created by Furong Huang on 9/25/13.
//  Copyright (c) 2013 Furong Huang. All rights reserved.
//

#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int NX;
int NA;
int KHID;
double alpha0;
int DATATYPE;
int main(int argc, const char * argv[])
{
	// number of nodes in the partitions
	NX = furong_atoi(argv[1]);
	NA = furong_atoi(argv[2]);
	KHID = furong_atoi(argv[3]);
	alpha0 = furong_atof(argv[4]);
	DATATYPE = furong_atoi(argv[5]);
	// 1500 12419 10 0 1	\
	// $(SolutionDir)\datasets\Nips\result\alpha.txt $(SolutionDir)\datasets\Nips\result\beta.txt $(SolutionDir)\datasets\Nips\result\hi.txt \
	// $(SolutionDir)\datasets\Nips\samples.txt 
	//3430  6906  10 0 1	$(SolutionDir)\datasets\Kos\result\alpha.txt $(SolutionDir)\datasets\Kos\result\beta.txt $(SolutionDir)\datasets\Kos\result\hi.txt $(SolutionDir)\datasets\Kos\samples.txt 
	const char* FILE_alpha_WRITE = argv[6];
	const char* FILE_beta_WRITE = argv[7];
	const char* FILE_hi_WRITE = argv[8];

	const char* FILE_GA = argv[9];

	TIME_start = clock();
	SparseMatrix<double> Gx_a(NX, NA);	Gx_a.resize(NX, NA);	
	Gx_a.makeCompressed();
	Gx_a = read_G_sparse((char *)FILE_GA, "GX_A", NX, NA);
	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices before preproc = %5.25e (Seconds)\n", time_readfile);



	cout << "(1) Whitening--------------------------" << endl;
	TIME_start = clock();
	SparseMatrix<double> W(NA, KHID); W.resize(NA, KHID); W.makeCompressed();	
	VectorXd mu_a(NA); 
	SparseMatrix<double> Uw(NA, KHID);  Uw.resize(NA, KHID); Uw.makeCompressed();
	SparseMatrix<double> diag_Lw_sqrt_inv_s(KHID, KHID); diag_Lw_sqrt_inv_s.resize(NA, KHID); diag_Lw_sqrt_inv_s.makeCompressed();
	VectorXd Lengths(NX);
	second_whiten_topic(Gx_a, W, mu_a, Uw, diag_Lw_sqrt_inv_s, Lengths);

	// whitened datapoints
	SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();	VectorXd Data_a_mu = W.transpose() * mu_a;

	TIME_end = clock();
	double time_whitening = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.25e (Seconds)\n", time_whitening);


	cout << "(2) Tensor decomposition----------------" << endl;
	TIME_start = clock();
	VectorXd lambda(KHID);
	MatrixXd phi_new(KHID, KHID);

	tensorDecom_alpha0_topic(Data_a_G, Data_a_mu, Lengths, lambda, phi_new);


	TIME_end = clock();
	double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.25e (Seconds)\n", time_stpm);


	cout << "(3) Unwhitening-----------" << endl;
	TIME_start = clock();
	MatrixXd Inv_Lambda = (pinv_vector(lambda)).asDiagonal();
	SparseMatrix<double> inv_lam_phi = (Inv_Lambda.transpose() * phi_new.transpose()).sparseView();
	SparseMatrix<double> pi_tmp1 = inv_lam_phi * W.transpose();
	VectorXd alpha(KHID);
	MatrixXd beta(NA, KHID);
	Unwhitening(lambda, phi_new, Uw, diag_Lw_sqrt_inv_s, alpha, beta);
	
	TIME_end = clock();
	double time_post = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for post processing = %5.25e (Seconds)\n", time_post);
	
	cout << "(4) Writing results----------" << endl;
	write_alpha((char *)FILE_alpha_WRITE, alpha);
	write_beta((char *)FILE_beta_WRITE, beta);


	// decode
	cout << "(5) Decoding-----------" << endl;
	TIME_start = clock();
	int inference = decode(alpha, beta, Lengths, Gx_a, (char*)FILE_hi_WRITE);
	TIME_end = clock();
	double time_decode = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for decoding = %5.25e (Seconds)\n", time_decode);


	cout << "(6) Program over------------" << endl;
	printf("\ntime taken for execution of the whole program = %5.25e (Seconds)\n", time_whitening + time_stpm + time_post);
	return 0;
}

