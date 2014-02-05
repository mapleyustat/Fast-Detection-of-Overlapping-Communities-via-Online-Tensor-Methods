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

#include "Community.h"

typedef unsigned long long timestamp_t;
typedef int bool_t;

// variables for measuring running time
timeval start_timeval_svd1, stop_timeval_svd1;
timeval start_timeval_svd2, stop_timeval_svd2;
timestamp_t measure_start_svd1, measure_stop_svd1; // timing for svd1
timestamp_t measure_start_svd2, measure_stop_svd2; // timing for svd2
timestamp_t measure_start_rd1, measure_stop_rd1; // timing for reading before pre processing
timestamp_t measure_start_rd2, measure_stop_rd2; // timing for reading after stochastic updates
timestamp_t measure_start_pre, measure_stop_pre; // timing for pre processing
timestamp_t measure_start_stpm, measure_stop_stpm; // timing for stochastic updates
timestamp_t measure_start_post, measure_stop_post; // timing for post processing
timestamp_t measure_start_error, measure_stop_error; // timing for error calculation
timeval start_timeval_pre, stop_timeval_pre;
timeval start_timeval_stpm, stop_timeval_stpm;
timeval start_timeval_post, stop_timeval_post;
timeval start_timeval_rd1, stop_timeval_rd1;
timeval start_timeval_rd2, stop_timeval_rd2;
timeval start_timeval_error, stop_timeval_error;

double time_pre, time_stpm, time_post; // time taken 
double time_svd1, time_svd2;
double time_rd1, time_rd2;
double time_error;


// main function
int main(int argc, const char * argv[])
{
  gettimeofday(&start_timeval_rd1, NULL);
  SparseMatrix<double> Gx_a(NX,NA);
  Gx_a.resize(NX,NA);
  SparseMatrix<double> Gx_b(NX,NB);
  Gx_b.resize(NX,NB);
  SparseMatrix<double> Gx_c(NX,NC);
  Gx_c.resize(NX,NC);
  Gx_a.makeCompressed();
  Gx_b.makeCompressed();
  Gx_c.makeCompressed();
  // reading the partitions
  Gx_a = read_G_sparse((char *) FILE_GA , "GX_A" ,NX, NA);
  Gx_b = read_G_sparse((char *) FILE_GB , "GX_B" ,NX, NB);
  Gx_c = read_G_sparse((char *) FILE_GC , "GX_C" ,NX, NC);

  gettimeofday(&stop_timeval_rd1, NULL);
  measure_stop_rd1 = stop_timeval_rd1.tv_usec + (timestamp_t)stop_timeval_rd1.tv_sec * 1000000;
  measure_start_rd1 = start_timeval_rd1.tv_usec + (timestamp_t)start_timeval_rd1.tv_sec * 1000000;
  time_rd1 = (measure_stop_rd1 - measure_start_rd1) / 1000000.0L;
  printf("Exec Time reading matrices before preproc = %5.25e (Seconds)\n",time_rd1);
  
  // initialize W, Z_B,Z_C, mu_a, mu_b, mu_c;
  SparseMatrix<double> W(NA,KHID); W.resize(NA,KHID); W.makeCompressed();
  SparseMatrix<double> Z_B(NA,NB); Z_B.resize(NA,NB); Z_B.makeCompressed();
  SparseMatrix<double> Z_C(NA,NC); Z_C.resize(NA,NC); Z_C.makeCompressed();
  VectorXd mu_a(NA);
  VectorXd mu_b(NB); 
  VectorXd mu_c(NC);
  
  cout << "----------------------------Before whitening--------------------------" << endl;
  gettimeofday(&start_timeval_pre, NULL);  // measuring start time for pre processing
  second_whiten(Gx_a,Gx_b,Gx_c,W,Z_B,Z_C,mu_a,mu_b,mu_c);
  // whitened datapoints
  SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();
  VectorXd Data_a_mu  = W.transpose() * mu_a;
  SparseMatrix<double> Data_b_G = W.transpose() * Z_B * Gx_b.transpose();
  VectorXd Data_b_mu  = W.transpose() * Z_B * mu_b;
  SparseMatrix<double> Data_c_G = W.transpose() * Z_C * Gx_c.transpose();
  VectorXd Data_c_mu  = W.transpose() * Z_C * mu_c;
  gettimeofday(&stop_timeval_pre, NULL);   // measuring stop time for pre processing
  
  measure_stop_pre = stop_timeval_pre.tv_usec + (timestamp_t)stop_timeval_pre.tv_sec * 1000000;
  measure_start_pre = start_timeval_pre.tv_usec + (timestamp_t)start_timeval_pre.tv_sec * 1000000;
  time_pre = (measure_stop_pre - measure_start_pre) / 1000000.0L;
  printf("time taken by preprocessing = %5.25e (Seconds)\n",time_pre);
    
  cout << "----------------------------After whitening---------------------------" << endl;
  // stochastic updates
  VectorXd lambda(KHID); 
  MatrixXd phi_new(KHID,KHID);
  cout << "------------------------------Before tensor decomposition----------------" << endl;
  gettimeofday(&start_timeval_stpm, NULL); // measuring start time for stochastic updates
  tensorDecom_alpha0(Data_a_G,Data_a_mu,Data_b_G,Data_b_mu,Data_c_G,Data_c_mu,lambda,phi_new);
  gettimeofday(&stop_timeval_stpm, NULL);  // measuring stop time for stochastic updates
  cout << "after tensor decomposition" << endl;
  measure_stop_stpm = stop_timeval_stpm.tv_usec + (timestamp_t)stop_timeval_stpm.tv_sec * 1000000;
  measure_start_stpm = start_timeval_stpm.tv_usec + (timestamp_t)start_timeval_stpm.tv_sec * 1000000;
  time_stpm = (measure_stop_stpm - measure_start_stpm) / 1000000.0L;
  cout << "------------------------------After tensor decomposition----------------" << endl;  
  printf("time taken by stochastic tensor decomposition = %5.25e (Seconds)\n",time_stpm);
  
  //cout <<  phi_new << endl; // optionally print eigenvectors
  cout << "the eigenvalues are:" << endl;
  cout << lambda << endl;
  
  
  // post processing
  cout << "------------Reading Gb_a, Gc_a---------"<<endl;
  gettimeofday(&start_timeval_rd2, NULL);
#ifdef CalErrALL
  // read the matrix Gab and Gac
  SparseMatrix<double> Gb_a(NB,NA);Gb_a.resize(NB,NA);
  SparseMatrix<double> Gc_a(NC,NA);Gc_a.resize(NC,NA);
  Gb_a = read_G_sparse((char *) FILE_Gb_a, "GB_A" ,NB, NA); Gb_a.makeCompressed();
  Gc_a = read_G_sparse((char *) FILE_Gc_a ,"GC_A" ,NC, NA); Gc_a.makeCompressed();
    // releasing memory of Gx_a, Gx_b, Gx_c;
    Gx_b.resize(0,0);Gx_c.resize(0,0);
#endif
  MatrixXd Inv_Lambda = (pinv_vector(lambda)).asDiagonal();
  SparseMatrix<double> inv_lam_phi = (Inv_Lambda.transpose() * phi_new.transpose()).sparseView();
    
  gettimeofday(&stop_timeval_rd2, NULL);
  measure_stop_rd2 = stop_timeval_rd2.tv_usec + (timestamp_t)stop_timeval_rd2.tv_sec * 1000000;
  measure_start_rd2 = start_timeval_rd2.tv_usec + (timestamp_t)start_timeval_rd2.tv_sec * 1000000;
  time_rd2 = (measure_stop_rd2 - measure_start_rd2) / 1000000.0L;
  cout << "------------After reading Gb_a, Gc_a---------"<<endl;
  printf("time taken for reading matrices after post processing = %5.25e (Seconds)\n",time_rd2);
  
  
  
  cout << "---------------------------Computing pi matrices-----------------------------" << endl;
  gettimeofday(&start_timeval_post, NULL);  // measuring start time for post processing
  
  SparseMatrix<double> pi_x(KHID,NX);pi_x.reserve(KHID*NX);pi_x.makeCompressed();
  SparseMatrix<double> pi_x_tmp1 = inv_lam_phi * W.transpose();
    
#ifdef CalErrALL
  SparseMatrix<double> pi_a(KHID,NA);pi_a.reserve(KHID*NA);pi_a.makeCompressed();
  SparseMatrix<double> pi_b(KHID,NB);pi_b.reserve(KHID*NB);pi_b.makeCompressed();
  SparseMatrix<double> pi_c(KHID,NC);pi_c.reserve(KHID*NC);pi_c.makeCompressed();
  
  pi_a = pi_x_tmp1 * Z_B * Gb_a;
  MatrixXd pi_a_full = (MatrixXd) pi_a;pi_a.resize(0,0);
    
  pi_b = pi_x_tmp1 * Gb_a.transpose();
  MatrixXd pi_b_full = (MatrixXd) pi_b;pi_b.resize(0,0);
    
  pi_c = pi_x_tmp1 * Gc_a.transpose();
  MatrixXd pi_c_full = (MatrixXd) pi_c;pi_c.resize(0,0);
#endif
    
  pi_x =pi_x_tmp1 * Gx_a.transpose();Gx_a.resize(0,0);
  MatrixXd pi_x_full = (MatrixXd) pi_x;pi_x.resize(0,0);
  gettimeofday(&stop_timeval_post, NULL);  // measuring stop time for post processing
  measure_stop_post = stop_timeval_post.tv_usec + (timestamp_t)stop_timeval_post.tv_sec * 1000000;
  measure_start_post = start_timeval_post.tv_usec + (timestamp_t)start_timeval_post.tv_sec * 1000000;
  time_post = (measure_stop_post - measure_start_post) / 1000000.0L;
  cout << "---------After post processing------------" << endl;
  printf("time taken for post processing = %5.25e (Seconds)\n",time_post);
  cout<<"-------------------------Concatenation for pi_est-------------------- "<< endl;
  
  // store true_pi
#ifdef CalErrALL
  long PI_LEN =(long) NX+NA+NB+NC;
#else
    long PI_LEN =(long) NX;
#endif
    
  MatrixXd My_pi_true_mat(KTRUE,PI_LEN);
  MatrixXd My_pi_est_mat(KHID,PI_LEN);
#ifdef CalErrALL
  for (int kk = 0; kk < KHID; kk++)
  {
    // for My_pi_est;
    VectorXd My_pi_est1(NX+NA);
    My_pi_est1 = concatenation_vector (pi_x_full.row(kk), pi_a_full.row(kk));
    VectorXd My_pi_est2(NX+NA+NB);
    My_pi_est2 =concatenation_vector (My_pi_est1, pi_b_full.row(kk));
    VectorXd My_pi_est3(NX+NA+NB+NC);
    My_pi_est3 =concatenation_vector (My_pi_est2, pi_c_full.row(kk));
    My_pi_est_mat.row(kk) = My_pi_est3;
  }
    pi_a_full.resize(0,0);
    pi_b_full.resize(0,0);
    pi_c_full.resize(0,0);
#else
    My_pi_est_mat =pi_x_full;
#endif
    pi_x_full.resize(0,0);
  
  // converting them to stochastic matrix
  My_pi_est_mat = normProbMatrix(My_pi_est_mat);
  SparseMatrix<double> sparse_my_pi_est_mat = My_pi_est_mat.sparseView();

  cout << "-----------Before writing results: W, Z_B,Z_C and pi-----------"<<endl;
  write_pi(FILE_PI_WRITE, sparse_my_pi_est_mat);
  write_pi(FILE_WHITE_WRITE, W);
  write_pi(FILE_INVLAMPHI_WRITE, inv_lam_phi);
  cout << "-----------After writing results---------"<< endl;
  
#ifdef ErrCal // set error calculation flag if it needs to be computed
  cout << "--------------------------------Calculating error------------------------------" << endl;
  gettimeofday(&start_timeval_error, NULL);  // measuring start time for error calculation
#ifdef CalErrALL
  // calculate error
  Gb_a.resize(0,0); Gc_a.resize(0,0);
  // read pi_true, i.e., ground truth matrices
  SparseMatrix<double> Pi_true_a(KTRUE,NA);Pi_true_a.makeCompressed();Pi_true_a = read_G_sparse((char *) FILE_Pi_a , "Pi_true_A" ,KTRUE, NA);
  MatrixXd Pi_true_a_full = (MatrixXd) Pi_true_a;  Pi_true_a.resize(0,0);
  SparseMatrix<double> Pi_true_b(KTRUE,NB);Pi_true_b.makeCompressed();Pi_true_b = read_G_sparse((char *) FILE_Pi_b , "Pi_true_B" ,KTRUE, NB);
  MatrixXd Pi_true_b_full = (MatrixXd) Pi_true_b;  Pi_true_b.resize(0,0);
  SparseMatrix<double> Pi_true_c(KTRUE,NC);Pi_true_c.makeCompressed();Pi_true_c = read_G_sparse((char *) FILE_Pi_c , "Pi_true_C" ,KTRUE, NC);
  MatrixXd Pi_true_c_full = (MatrixXd) Pi_true_c;  Pi_true_c.resize(0,0);
#endif
  SparseMatrix<double> Pi_true_x(KTRUE,NX);Pi_true_x.makeCompressed();Pi_true_x = read_G_sparse((char *) FILE_Pi_x , "Pi_true_X" ,KTRUE, NX);
  MatrixXd Pi_true_x_full = (MatrixXd) Pi_true_x;  Pi_true_x.resize(0,0);
  
  /*
  // this is only for yelp, comment this for DBLP
  long PI_LEN = (long)NX;
  MatrixXd My_pi_true_mat(KTRUE,PI_LEN);
  My_pi_true_mat =  Pi_true_x_full;
  MatrixXd My_pi_est_mat(KHID,PI_LEN); 
  My_pi_est_mat = pi_x_full;
  */    
  
  cout<<"-------------------------Concatenation for pi_true-------------------- "<< endl;
#ifdef CalErrALL
  for ( int k = 0; k < KTRUE; k++)
  {
    // for My_pi_true;
    VectorXd My_pi_true1(NX+NA);
    My_pi_true1 = concatenation_vector ((Pi_true_x_full.row(k)),(Pi_true_a_full.row(k)));
    VectorXd My_pi_true2(NX+NA+NB);
    My_pi_true2 =concatenation_vector (My_pi_true1, (Pi_true_b_full.row(k)));
    VectorXd My_pi_true3(NX+NA+NB+NC);
    My_pi_true3 =concatenation_vector (My_pi_true2, (Pi_true_c_full.row(k)));
    My_pi_true_mat.row(k) = My_pi_true3;
  } 
  Pi_true_a_full.resize(0,0);
  Pi_true_b_full.resize(0,0);
  Pi_true_c_full.resize(0,0);
#else
    My_pi_true_mat = Pi_true_x_full;
#endif
  Pi_true_x_full.resize(0,0);
    
    
    double thresh_vec[NUM_THRESH] = thresh_vec_def;
    //{0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1,0.08};
  double error_vec[NUM_THRESH] = {0.0};
  double match_vec[NUM_THRESH] = {0.0};
  
  for (int tttt = 0; tttt < NUM_THRESH; tttt++)
  {
    MatrixXd p_values=MatrixXd::Zero(KTRUE,KHID);
    MatrixXd errors=MatrixXd::Zero(KTRUE,KHID);	  
    for ( int k = 0; k < KTRUE; k++)
    {
      VectorXd my_pi_true_eigen = My_pi_true_mat.row(k);
      double *my_pi_true = my_pi_true_eigen.data();
      for (int kk = 0; kk < KHID; kk++)
      {
	VectorXd my_pi_est_eigen = My_pi_est_mat.row(kk);
	double *my_pi_est = my_pi_est_eigen.data();
	
	for(long lltt = 0; lltt < PI_LEN; lltt++)
	{
	  if(my_pi_est[lltt] < thresh_vec[tttt])
	    my_pi_est[lltt] = 0;
	  else
	    my_pi_est[lltt] = 1;
	}
	// calculate p-values and error
	double correlation = Calculate_Correlation(my_pi_est, my_pi_true, (long)PI_LEN); //{long}
	if (correlation > 0)
	{
	  p_values(k,kk)=Calculate_Pvalue(my_pi_true, my_pi_est, (long)PI_LEN); //{long}
	  if (p_values(k,kk) < PVALUE_TOLE)
	  {
	    errors(k,kk)=(my_pi_true_eigen - my_pi_est_eigen).cwiseAbs().sum();
	  }
	  else
	  {
	    errors(k,kk)=0;
	  }
	}
	else
	{
	  p_values(k,kk)=-1;
	  errors(k,kk)=0;
	}
      }
    }
    VectorXd matched = errors.rowwise().sum();
    double nnz =0;
    for(long calc=0; calc <KTRUE; calc++)
    {
      if(matched(calc)>0)
	nnz++;
    }
    error_vec[tttt]=(double)errors.sum()/((double)PI_LEN*KTRUE);
    match_vec[tttt]=((double)nnz)/((double)KTRUE);
  }
  gettimeofday(&stop_timeval_error, NULL);  // measuring stop time for error calculation
  measure_stop_error = stop_timeval_error.tv_usec + (timestamp_t)stop_timeval_error.tv_sec * 1000000;
  measure_start_error = start_timeval_error.tv_usec + (timestamp_t)start_timeval_error.tv_sec * 1000000;
  time_error = (measure_stop_error - measure_start_error) / 1000000.0L;
  cout << "---------After error calculation------------"<<endl;
  printf("time taken for error calculation = %5.25e (Seconds)\n",time_error);
  
  furongprintVector(thresh_vec, NUM_THRESH, "thresh vector "); // outputs are printed
  furongprintVector(error_vec, NUM_THRESH, "error vector ");
  furongprintVector(match_vec, NUM_THRESH, "match vector ");
#endif
  cout << "Program over" << endl;    
  printf("\ntime taken for execution of the whole program = %5.25e (Seconds)\n", time_rd1 + time_pre + time_stpm + time_rd2 + time_post);
  return 0;
}
