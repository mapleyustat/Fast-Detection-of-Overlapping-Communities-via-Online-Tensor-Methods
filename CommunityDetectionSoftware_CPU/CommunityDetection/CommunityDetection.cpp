// CommunityDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
using namespace Eigen;
using namespace std;
clock_t TIME_start, TIME_end;
int NX;
int NA;
int NB;
int NC;
int KHID;
double alpha0;
int DATATYPE;
int main(int argc, const char * argv[])
{
	// 65879 65879 65879 65880 10 0.01 0

	NX = furong_atoi(argv[1]);
	NA = furong_atoi(argv[2]);
	NB = furong_atoi(argv[3]);
	NC = furong_atoi(argv[4]);
	KHID = furong_atoi(argv[5]);
	alpha0 = furong_atof(argv[6]);
	DATATYPE = furong_atoi(argv[7]);

	// $(SolutionDir)\data\Gx_a.txt $(SolutionDir)\data\Gx_b.txt $(SolutionDir)\data\Gx_c.txt $(SolutionDir)\data\Gb_a.txt $(SolutionDir)\data\Gc_a.txt 

	const char* FILE_GA = argv[8];
	const char* FILE_GB = argv[9];
	const char* FILE_GC = argv[10];
	const char* FILE_Gb_a = argv[11];
	const char* FILE_Gc_a = argv[12];

	//$(SolutionDir)\data\result\pi.txt $(SolutionDir)\data\result\white.txt  

	const char* FILE_PI_WRITE = argv[13];
	const char* FILE_WHITE_WRITE = argv[14];
//	const char* FILE_INVLAMPHI_WRITE = argv[15];

	
	cout << "(1) Reading Graph Partitions----" << endl;
	TIME_start = clock();
	SparseMatrix<double> Gx_a(NX, NA);	Gx_a.resize(NX, NA);	SparseMatrix<double> Gx_b(NX, NB);	Gx_b.resize(NX, NB);	SparseMatrix<double> Gx_c(NX, NC);	Gx_c.resize(NX, NC);	Gx_a.makeCompressed();	Gx_b.makeCompressed();	Gx_c.makeCompressed();
	// reading the partitions
	Gx_a = read_G_sparse((char *)FILE_GA, "GX_A", NX, NA);
	Gx_b = read_G_sparse((char *)FILE_GB, "GX_B", NX, NB);
	Gx_c = read_G_sparse((char *)FILE_GC, "GX_C", NX, NC);
	TIME_end = clock();
	double time_readfile = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("Exec Time reading matrices = %5.25e (Seconds)\n", time_readfile);


	
	cout << "(2) Whitening----" << endl;
	TIME_start = clock();
	SparseMatrix<double> W(NA, KHID); W.resize(NA, KHID); W.makeCompressed();	
	SparseMatrix<double> Z_B_part1(NA, KHID); SparseMatrix<double>Z_B_part2(KHID, NB); Z_B_part1.resize(NA, KHID); Z_B_part2.resize(KHID, NB);
	SparseMatrix<double> Z_C_part1(NA, KHID); SparseMatrix<double>Z_C_part2(KHID, NC); Z_C_part1.resize(NA, KHID); Z_C_part2.resize(KHID, NC);
	VectorXd mu_a(NA);	VectorXd mu_b(NB);	VectorXd mu_c(NC);
	
	second_whiten(Gx_a, Gx_b, Gx_c, W, Z_B_part1,Z_B_part2, Z_C_part1, Z_C_part2, mu_a, mu_b, mu_c);

	// whitened datapoints
	SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();	VectorXd Data_a_mu = W.transpose() * mu_a;
	SparseMatrix<double> Data_b_G = (W.transpose() * Z_B_part1)* Z_B_part2 * Gx_b.transpose();	VectorXd Data_b_mu = (W.transpose() * Z_B_part1)* Z_B_part2 * mu_b;
	SparseMatrix<double> Data_c_G = (W.transpose() * Z_C_part1)* Z_C_part2 * Gx_c.transpose();	VectorXd Data_c_mu = (W.transpose() * Z_C_part1)* Z_C_part2 * mu_c;
	TIME_end = clock();
	double time_whitening = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.25e (Seconds)\n", time_whitening);

	
	cout << "(3) Tensor decomposition---------" << endl;
	TIME_start = clock();
	VectorXd lambda(KHID);
	MatrixXd phi_new(KHID, KHID);

	tensorDecom_alpha0(Data_a_G, Data_a_mu, Data_b_G, Data_b_mu, Data_c_G, Data_c_mu, lambda, phi_new);
	// releasing memory of Gx_a, Gx_b, Gx_c;
	Gx_b.resize(0, 0); Gx_c.resize(0, 0);

	TIME_end = clock();
	double time_stpm = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken by whitening = %5.25e (Seconds)\n", time_stpm);


	
	cout << "(4) Postprocessing to get membership vectors for individuals---------" << endl;
	TIME_start = clock();
	// read the matrix Gab and Gac
	SparseMatrix<double> Gb_a(NB, NA); Gb_a.resize(NB, NA);
	SparseMatrix<double> Gc_a(NC, NA); Gc_a.resize(NC, NA);
	Gb_a = read_G_sparse((char *)FILE_Gb_a, "GB_A", NB, NA); Gb_a.makeCompressed();
	Gc_a = read_G_sparse((char *)FILE_Gc_a, "GC_A", NC, NA); Gc_a.makeCompressed();
	SparseMatrix<double> pi_x(KHID, NX); pi_x.reserve(KHID*NX); pi_x.makeCompressed();	SparseMatrix<double> pi_a(KHID, NA); pi_a.reserve(KHID*NA); pi_a.makeCompressed();	SparseMatrix<double> pi_b(KHID, NB); pi_b.reserve(KHID*NB); pi_b.makeCompressed();	SparseMatrix<double> pi_c(KHID, NC); pi_c.reserve(KHID*NC); pi_c.makeCompressed();

	MatrixXd Inv_Lambda = (pinv_vector(lambda)).asDiagonal();
	SparseMatrix<double> inv_lam_phi = (Inv_Lambda.transpose() * phi_new.transpose()).sparseView();
	SparseMatrix<double> pi_tmp1 = inv_lam_phi * W.transpose();
	pi_a = (pi_tmp1* Z_B_part1)* Z_B_part2 * Gb_a;	MatrixXd pi_a_dense = (MatrixXd)pi_a; pi_a.resize(0, 0);

	pi_b = pi_tmp1 * Gb_a.transpose();	MatrixXd pi_b_dense = (MatrixXd)pi_b; pi_b.resize(0, 0);

	pi_c = pi_tmp1 * Gc_a.transpose();	MatrixXd pi_c_dense = (MatrixXd)pi_c; pi_c.resize(0, 0);


	pi_x = pi_tmp1 * Gx_a.transpose(); Gx_a.resize(0, 0);	MatrixXd pi_x_dense = (MatrixXd)pi_x; pi_x.resize(0, 0);

	TIME_end = clock();
	double time_post = double(TIME_end - TIME_start) / CLOCKS_PER_SEC;
	printf("time taken for post processing = %5.25e (Seconds)\n", time_post);
//	cout << "-----------pi_x_dense:nonZeros()" << pi_x_dense.nonZeros() << "-------------" << endl;
//	cout << "-----------pi_a_dense:nonZeros()" << pi_a_dense.nonZeros() << "-------------" << endl;
//	cout << "-----------pi_b_dense:nonZeros()" << pi_b_dense.nonZeros() << "-------------" << endl;
//	cout << "-----------pi_c_dense:nonZeros()" << pi_c_dense.nonZeros() << "-------------" << endl;

	cout << "(5) Concatenating pi_est-------------------- " << endl;

	// store est_pi
	long PI_LEN = (long)NX + NA + NB + NC;
	MatrixXd My_pi_est_mat(KHID, PI_LEN);
	for (int kk = 0; kk < KHID; kk++)
	{
		VectorXd My_pi_est1(NX + NA);
		My_pi_est1 = concatenation_vector(pi_x_dense.row(kk), pi_a_dense.row(kk));
		VectorXd My_pi_est2(NX + NA + NB);
		My_pi_est2 = concatenation_vector(My_pi_est1, pi_b_dense.row(kk));
		VectorXd My_pi_est3(NX + NA + NB + NC);
		My_pi_est3 = concatenation_vector(My_pi_est2, pi_c_dense.row(kk));
		My_pi_est_mat.row(kk) = My_pi_est3;
	}
	pi_a_dense.resize(0, 0);
	pi_b_dense.resize(0, 0);
	pi_c_dense.resize(0, 0);
	pi_x_dense.resize(0, 0);

	// converting them to stochastic matrix
	My_pi_est_mat = normProbMatrix(My_pi_est_mat);
	SparseMatrix<double> sparse_my_pi_est_mat = My_pi_est_mat.sparseView();

	cout << "(6) Writing results: W, Z_B,Z_C and pi-----------" << endl;
	write_pi((char *) FILE_PI_WRITE, sparse_my_pi_est_mat);
	write_pi((char *) FILE_WHITE_WRITE, W);

	cout << "Program over" << endl;
	printf("\ntime taken for execution of the whole program = %5.25e (Seconds)\n", time_whitening + time_stpm  + time_post);
	return 0;
}

