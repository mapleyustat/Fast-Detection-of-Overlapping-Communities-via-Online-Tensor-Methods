//
//  main.cpp
//  Community
//
//  Created by Furong Huang on 9/25/13.
//  Copyright (c) 2013 Furong Huang. All rights reserved.
//

#include "Community.h"

typedef unsigned long long timestamp_t;
typedef int bool_t;

timeval start_timeval_svd1, stop_timeval_svd1;
timeval start_timeval_svd2, stop_timeval_svd2;
timestamp_t measure_start_svd1, measure_stop_svd1;       // Timing for svd1
timestamp_t measure_start_svd2, measure_stop_svd2;       // Timing for svd2
timestamp_t measure_start_rd1, measure_stop_rd1;       // Timing for reading before pre_proc 
timestamp_t measure_start_rd2, measure_stop_rd2;       // Timing for reading after stpm
timestamp_t measure_start_pre, measure_stop_pre;       // Timing for pre_proc 
timestamp_t measure_start_stpm, measure_stop_stpm;     // Timing for stpm
timestamp_t measure_start_post, measure_stop_post;     // Timing for post_proc
timestamp_t measure_start_error, measure_stop_error;     // Timing for error_calc
timeval start_timeval_pre, stop_timeval_pre;
timeval start_timeval_stpm, stop_timeval_stpm;
timeval start_timeval_post, stop_timeval_post;
timeval start_timeval_rd1, stop_timeval_rd1;
timeval start_timeval_rd2, stop_timeval_rd2;
timeval start_timeval_error, stop_timeval_error;

double time_pre, time_stpm, time_post;                 // Time taken 
double time_svd1, time_svd2;
double time_rd1, time_rd2;
double time_error;


int main(int argc, const char * argv[])
{
  /*
  SparseMatrix<double> gxa(250, 250);
  gxa.reserve(250*250);
  gxa.makeCompressed();

  gxa = read_G_sparse("./sparse000/sparseGx_a.txt", "gxa", 250, 250);
  cout << "read over" << endl;
  write_pi("out_piiiiiii.txt", gxa);
  exit(0);
*/
  gettimeofday(&start_timeval_rd1, NULL);
    SparseMatrix<double> Gx_a(NX,NA);
    Gx_a.resize(NX,NA);
#ifdef CommunityModel
    SparseMatrix<double> Gx_b(NX,NB);
    Gx_b.resize(NX,NB);
    SparseMatrix<double> Gx_c(NX,NC);
    Gx_c.resize(NX,NC);
    Gx_a.makeCompressed();
    Gx_b.makeCompressed();
    Gx_c.makeCompressed();
#endif
    //********* read Gx_a;
    Gx_a = read_G_sparse((char *) FILE_GA , "GX_A" ,NX, NA);
#ifdef CommunityModel
    Gx_b = read_G_sparse((char *) FILE_GB , "GX_B" ,NX, NB);
    Gx_c = read_G_sparse((char *) FILE_GC , "GX_C" ,NX, NC);
#endif
    /*
    srand(clock());
    for (int i = 0; i< NX; i+=10)
      {
	//	for(int j =0; j<min(min(NA,NB),NC); j++)
	//	  {
	    // double randnum = rand()%3;
	// VectorXd randnum10set(10);
	    for (int j =0; j <10; j++)
	      {
	      Gx_a.coeffRef(i,rand()%min(min(NA,NB),NC))=1;
	       Gx_b.coeffRef(i,rand()%min(min(NA,NB),NC))=1;
	       Gx_c.coeffRef(i,rand()%min(min(NA,NB),NC))=1;
	      }
	   
	    // Gx_a.coeffRef(i,j)=1;
	    // if (rand()%2 == 0)
	    //    Gx_b.coeffRef(i,j)=1;
	    // if (rand()%2 == 0)
	    // Gx_c.coeffRef(i,j)=1;
	    // //	  }
      }
    */
    //********* excution time
    gettimeofday(&stop_timeval_rd1, NULL);
     measure_stop_rd1 = stop_timeval_rd1.tv_usec + (timestamp_t)stop_timeval_rd1.tv_sec * 1000000;
     measure_start_rd1 = start_timeval_rd1.tv_usec + (timestamp_t)start_timeval_rd1.tv_sec * 1000000;
     time_rd1 = (measure_stop_rd1 - measure_start_rd1) / 1000000.0L;
     printf("Exec Time reading matrices before preproc = %5.25e (Seconds)\n",time_rd1);
       
     //********* Initialize W, Z_B,Z_C, mu_a, mu_b, mu_c;
     SparseMatrix<double> W(NA,KHID); W.resize(NA,KHID); W.makeCompressed();
#ifdef CommunityModel
     SparseMatrix<double> Z_B(NA,NB); Z_B.resize(NA,NB); Z_B.makeCompressed();
     SparseMatrix<double> Z_C(NA,NC); Z_C.resize(NA,NC); Z_C.makeCompressed();
#endif
    VectorXd mu_a(NA);
#ifdef CommunityModel
    VectorXd mu_b(NB); 
    VectorXd mu_c(NC);
#endif
   
    cout << "----------------------------Before whitening--------------------------" << endl;
    gettimeofday(&start_timeval_pre, NULL);  // Measuring start time for pre processing
#ifdef CommunityModel
    second_whiten(Gx_a,Gx_b,Gx_c,W,Z_B,Z_C,mu_a,mu_b,mu_c);
#else
    second_whiten_topic(Gx_a,W,mu_a);
#endif
        //**********  Whitened datapoints
    SparseMatrix<double> Data_a_G = W.transpose() * Gx_a.transpose();
    VectorXd Data_a_mu  = W.transpose() * mu_a;
#ifdef CommunityModel
    SparseMatrix<double> Data_b_G = W.transpose() * Z_B * Gx_b.transpose();
    VectorXd Data_b_mu  = W.transpose() * Z_B * mu_b;
    SparseMatrix<double> Data_c_G = W.transpose() * Z_C * Gx_c.transpose();
    VectorXd Data_c_mu  = W.transpose() * Z_C * mu_c;
#endif
    gettimeofday(&stop_timeval_pre, NULL);   // Measuring stop time for pre processing
    cout << "----------------------------END of  whitening---------------------------" << endl;
    measure_stop_pre = stop_timeval_pre.tv_usec + (timestamp_t)stop_timeval_pre.tv_sec * 1000000;
    measure_start_pre = start_timeval_pre.tv_usec + (timestamp_t)start_timeval_pre.tv_sec * 1000000;
    time_pre = (measure_stop_pre - measure_start_pre) / 1000000.0L;
    printf("Exec Time Pre Proc = %5.25e (Seconds)\n",time_pre);
    //cout << "-----------mu_a---------:\n" <<mu_a<<endl;
    //cout << "-----------mu_b---------:\n" << mu_b<<endl;
    //cout << "----------mu_c----------:\n"<< mu_c<<endl;

    
    //**********  Stochastic updates
    VectorXd lambda(KHID); 
    MatrixXd phi_new(KHID,KHID);
    cout << "------------------------------Before tensor decomposition----------------" << endl;
    gettimeofday(&start_timeval_stpm, NULL);
    //
#ifdef CommunityModel
    tensorDecom_alpha0(Data_a_G,Data_a_mu,Data_b_G,Data_b_mu,Data_c_G,Data_c_mu,lambda,phi_new);
#else
    tensorDecom_alpha0_topic(Data_a_G,Data_a_mu,lambda,phi_new);
#endif 
    gettimeofday(&stop_timeval_stpm, NULL); 
    cout << "after tensor decomposition" << endl;
    measure_stop_stpm = stop_timeval_stpm.tv_usec + (timestamp_t)stop_timeval_stpm.tv_sec * 1000000;
    measure_start_stpm = start_timeval_stpm.tv_usec + (timestamp_t)start_timeval_stpm.tv_sec * 1000000;
    time_stpm = (measure_stop_stpm - measure_start_stpm) / 1000000.0L;
   
    cout << "-------------------------TENSOR DECOM OVER---------------" << endl;
     printf("Exec Time STPM = %5.25e (Seconds)\n",time_stpm);

    //cout <<  phi_new << endl;
    cout << " PRINTING EVAL" << endl;
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

#ifdef CommunityModel
  
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
#else
  
  SparseMatrix<double> pi_x(KHID,NX);pi_x.reserve(KHID*NX);pi_x.makeCompressed();
  cout << "inv_lam_phi.rows()"<< inv_lam_phi.rows() << ";inv_lam_phi.cols()"<< inv_lam_phi.cols()<< endl;
  SparseMatrix<double> pi_x_tmp1 = inv_lam_phi * W.transpose();

  pi_x = pi_x_tmp1;

#endif
    
 
  MatrixXd pi_x_full = (MatrixXd) pi_x;pi_x.resize(0,0);
  gettimeofday(&stop_timeval_post, NULL);  // measuring stop time for post processing
  measure_stop_post = stop_timeval_post.tv_usec + (timestamp_t)stop_timeval_post.tv_sec * 1000000;
  measure_start_post = start_timeval_post.tv_usec + (timestamp_t)start_timeval_post.tv_sec * 1000000;
  time_post = (measure_stop_post - measure_start_post) / 1000000.0L;
  cout << "---------After post processing------------" << endl;
  printf("time taken for post processing = %5.25e (Seconds)\n",time_post);
  cout<<"-------------------------Concatenation for pi_est-------------------- "<< endl;
  
  // store true_pi
#ifdef CommunityModel
#ifdef CalErrALL
  long PI_LEN =(long) NX+NA+NB+NC;
#else
    long PI_LEN =(long) NX;
#endif
#else
    long PI_LEN = (long) NA;
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
#ifdef CommunityModel
      My_pi_est_mat = normProbMatrix(My_pi_est_mat);
#endif
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
    
  //  double thresh_vec[NUM_THRESH] = {0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1,0.08};
  double thresh_vec[NUM_THRESH] = thresh_vec_def;

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
