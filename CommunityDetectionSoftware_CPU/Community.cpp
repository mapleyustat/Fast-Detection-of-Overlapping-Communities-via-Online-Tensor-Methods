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

timeval start_timeval_svd_m, stop_timeval_svd_m;
timestamp_t measure_start_svd_m, measure_stop_svd_m; // timing for svdm
double time_svd_m;

// writing pi matrices
int write_pi(char *filename, SparseMatrix<double> mat)
{
  fstream f(filename, ios::out);
  for (long k=0; k<mat.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
      {
	f << it.row()+1 << "\t" << it.col()+1 << "\t" << it.value() << endl;
      }
  f.close();
  return 0;
}

// Furong's function for calculating p-values
double CalculateMean(double value[], long len) // calculating sample mean of an array of values
{
    double sum=0;
    for (long i=0; i<len; i++)
    {
        sum += value[i];        
    }
    return (double)(sum / len);
}

double CalculateVariance(double value[], long len) // calculating variance of an array of values
{
    double mean = CalculateMean(value, len);
    double temp =0;
    for (long i=0; i < len; i++)
    {
        temp +=(value[i]-mean)*(value[i]-mean);
    }
    return (double)(temp / len);
}

double CalculateSampleVariance(double value[], long len) // calculating sample variance of an array of values
{
    double mean = CalculateMean(&*value, len);    
    double temp =0;
    for (long i=0; i < len; i++)
    {
        temp +=(value[i]-mean)*(value[i]-mean);
    }
    return (double)(temp / (len-1));
    
}

double Calculate_StandardDeviation(double value[], long len) // calculating standard deviation of an array of values
{
    return sqrt(CalculateVariance(&*value, len));
}

double Calculate_SampleStandardDeviation(double value[], long len) // calculating sample standard deviation of an array of values
{
    return sqrt(CalculateSampleVariance(&*value, len));
}

double Calculate_Covariance(double x[], double y[], long len) // calculating cross-covariance of two arrays of values
{
    double x_mean=CalculateMean(x, len);
    double y_mean=CalculateMean(y, len);
    double summation =0;
    for (long i=0; i<len;i++)
    {
        summation += (x[i]-x_mean)*(y[i]-y_mean);
    }
    return (double)(summation /len);
}

double Calculate_Correlation(double x[], double y[], long len) // calculating correlation coefficient between two arrays of values
{
    double covariance = Calculate_Covariance(&*x, &*y, len);
    double correlation = covariance / (Calculate_StandardDeviation(&*x, len)*Calculate_StandardDeviation(&*y,len));
    return (correlation);
}

double Calculate_Tstat(double x[], double y[], long len) // calculating t-statistic for hypothesis testing
{
    double r = Calculate_Correlation(&*x, &*y, len);
    double t = r * sqrt((len-2)/(1-r*r));
    return t;
}

// FUNCTION betacf(a,b,x)
double betacf(double a, double b, double x)
{
    double betacfvalue;
    int MAXIT=100000;
    double EPS_thisfunc=3e-7;
    double FPMIN=1e-30;
    int m,m2;
    double aa,c,d,del,h,qab,qam,qap;
    qab=a+b;
    qap=a+1.0;
    qam=a-1.0;
    c=1.0;
    d=1.0-qab*x/qap;
    if (fabs(d)<FPMIN)
    {
        d=FPMIN;
    }
    
    d=1.0/d;
    h=d;
    m=0;
    do{
        m += 1;
        m2 = 2*m;
        aa = m*(b-m)*x/((qam+m2)*(a+m2));
        d = 1.0+aa*d;
        if(fabs(d) < FPMIN)
        {
            d = FPMIN;
        }
        c = 1.0+aa/c;
        if(fabs(c) < FPMIN)
        {
            c=FPMIN;
        }
        d=1.0/d;
        h=h*d*c;
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
        d=1.0+aa*d;
        if(fabs(d)<FPMIN)
        {
            d=FPMIN;
        }
        c=1.0+aa/c;
        if(fabs(c)<FPMIN)
        {
            c=FPMIN;
            
        }
        d=1.0/d;
        del=d*c;
        h=h*del;
        double theta=fabs(del-1.0);
        if(theta<EPS_thisfunc || theta==EPS_thisfunc)
        {
            goto STOP;
        }
    }while(m<=MAXIT);
    
    printf("a or b too big or MAXIT too small in betacf\n");
    printf("Redo: \n");
    system("pause");
    betacfvalue = betacf(a, b, x);
STOP: betacfvalue=h;
    return betacfvalue;
}

double gammln(double xx)
{
    double x,tmp,ser;
    static double cof[6]={76.18009173,-86.50532033,24.01409822,
        -1.231739516,0.120858003e-2,-0.536382e-5};
    int j;
    
    x=xx-1.0;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.0;
    for (j=0;j<=5;j++) {
        x += 1.0;
        ser += cof[j]/x;
    }
    return -tmp+log(2.50662827465*ser);
}

double betainc(double a, double b, double x)
{
    double betaivalue,bt;
    if (x == 0 || x > 1)
    {
        sleep(2);
        printf("bad argument x in betainc");
    }
    if (x == 0 || x == 1)
    {
        bt=0;
    }
    else
    {
        bt=exp(gammln(a+b)-gammln(a)-gammln(b) +a*log(x)+b*log(1.0-x));
    }
    if(x<(a+1.0)/(a+b+2.0))
    {
        betaivalue=bt * betacf(a,b,x)/a;
        return betaivalue;
    }
    else
    {
        betaivalue=1.0-bt * betacf(b,a,1.0-x)/b;
        return betaivalue;
    }
}

double Calculate_Pvalue(double x[], double y[], long len) // calculating p values
{
    double p;
    double t = -fabs(Calculate_Tstat(&*x, &*y, len));
    double n = (double)(len - 2);
    double normcutoff = 1e7;
    // Initialize P.
	p = NAN;
    bool nans =(isnan(t)|| n<=0);
    if (n==1)
    {
        p = 0.5 + atan(t)/M_PI;
        //return (2*p);
    }
    if ( n > normcutoff)
    {
        p= 0.5 * erfc(-t/ sqrt(2.0));
        //return (2*p);
    }
    if (n != 1 && n <= normcutoff && !nans)
    {
        double    temptemp = n / (n + t*t);
        p = 0.5* betainc (0.5*n,0.5,temptemp);
        if ( t >0 )
        {
            p=1-p;
            //return (2*p);
        }
        
    }
    else
    {
        p=0.5;
    }
    return (2*p);
}
/////////////////
void furongprintVector(double value[], long len, char *character) // print the elements of an array
{
    for(long i=0; i<len; i++)
    {
        printf("%s=%.10f\n",character,value[i]);
    }
}
////////////////////////////////////////////////////
// set of svd functions
//////////
std::pair<Eigen::MatrixXd, Eigen::VectorXd> k_svd (Eigen::MatrixXd A, int k) // selecting top k singular values and singular vectors from eigen dense svd - returns left singular vectors and singular values
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd u = svd.matrixU();
    Eigen::VectorXd s = svd.singularValues();
    cout << "singular values:\n" << s<< endl;
    pair<Eigen::MatrixXd, Eigen::VectorXd> mv;
    mv.first = u.leftCols(k);
    mv.second = s.head(k);
    return mv;
}
///////////////////////////////////////////
std::pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> k_svd_observabe (Eigen::MatrixXd A, int k) // same as k_svd function but without the printing
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd v = svd.matrixV();
    Eigen::VectorXd s = svd.singularValues();
    pair<pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::VectorXd> mv;
    mv.first.first = u.leftCols(k);
    mv.first.second = v.leftCols(k);
    mv.second = s.head(k);
    return mv;
}
std::pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > k_svd_observabe (Eigen::SparseMatrix<double> A, int k) // k svd of the observable nodes in sparse format using the asymmetric nystrom method
{
    pair<pair<SparseMatrix<double>,SparseMatrix<double> >, SparseVector<double> > u_l =SVD_asymNystrom_sparse(A,k);
    //    EigenSparseSVD u_l = sparse_svd(A,k);
    Eigen::SparseMatrix<double> u = u_l.first.first;
    Eigen::SparseMatrix<double> v = u_l.first.second;
    Eigen::SparseVector<double> s = u_l.second;
    pair<pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double> >, Eigen::SparseVector<double> > mv;
    mv.first.first.resize((int)u.rows(), (int)u.cols());
    mv.first.second.resize((int)v.rows(), (int)v.cols());
    mv.second.resize((int)s.size());
    mv.first.first = u;
    mv.first.second = v;
    mv.second = s;
    return mv;
}

/////////////////////// svd tested
pair<pair<Eigen::MatrixXd,Eigen::MatrixXd>, Eigen::VectorXd> latenttree_svd (Eigen::MatrixXd A)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd u = svd.matrixU();
    Eigen::VectorXd s = svd.singularValues();
    Eigen::MatrixXd v = svd.matrixV();
    pair<pair<Eigen::MatrixXd,Eigen::MatrixXd>, Eigen::VectorXd> mv;
    mv.first.first  = u;
    mv.first.second =v;
    mv.second = s;
    return mv;
}
//////////////////////////////////
//Nystrom for sparseSVD
/////////////////////////
SparseMatrix<double> random_embedding_mat(long cols_num, int k) // compute the random embedding using a diagonal matrix with iid rademacher entries
{
    cout<<"-----------begining of random_embeding_mat---------------"<<endl;
    srand(clock());
    SparseMatrix<double> phi(cols_num,k);
    for(int i = 0; i<cols_num; i++)
    {
       
        int r = rand()%k;// randomly-------
	int r_sign = rand()%5;
        double p_rand = (double)r_sign/(double)5.0;
        if (p_rand<0.5)
        phi.coeffRef(i,r)=-1;
    }
    phi.makeCompressed();
    phi.prune(EPS);
    return phi;
}
///////////////////////////////////////////////////////////////////////////////
pair< SparseMatrix<double>, SparseVector<double> > SVD_symNystrom_sparse(SparseMatrix<double> A,int k)
{
    pair< SparseMatrix<double>, SparseVector<double> > result;
    //////////////////////////////////////
    SparseMatrix<double> random_mat = random_embedding_mat((long)A.cols(),k);
    SparseMatrix<double> C=A*random_mat;
    C.makeCompressed(); C.prune(EPS);
    // QR of C
    SparseQR<SparseMatrix<double>,NaturalOrdering<int> > Q_R(C);
    // R
    MatrixXd R = (MatrixXd)(Q_R.matrixR()).topLeftCorner(k, k);
    MatrixXd R_inv = pinv_matrix(R);
    SparseMatrix<double> R_inv_sparse = R_inv.sparseView();
    // Q
    //SparseMatrix<double> Q;
    // First alternative
    
   
    // Anothe option which might work if sparse_svd gives errors, this might be even faster ..
    SparseMatrix<double> Q_tmp = C * Q_R.colsPermutation();
    SparseMatrix<double> Q = Q_tmp * R_inv_sparse;
    // Q.makeCompressed();
    //////

    Q.makeCompressed();
    Q.prune(EPS);
    // W
    MatrixXd W = (MatrixXd)random_mat.transpose() * C;
    MatrixXd W_sqrt = sqrt_matrix(W);
    SparseMatrix<double> W_sqrt_sparse = W_sqrt.sparseView();
    // getting column span u
    SparseMatrix<double> u_tmp = Q * R_inv_sparse.transpose();
    SparseMatrix<double> u = u_tmp * W_sqrt_sparse.transpose();
    u.makeCompressed(); u.prune(EPS);
    // orthogolize u
    SparseQR<SparseMatrix<double>,NaturalOrdering<int> > u_Q_R(u);
    SparseMatrix<double> u_Q; SparseMatrix<double> u_R_inv_sparse;
    MatrixXd u_R_dense = (MatrixXd) u_Q_R.matrixR().topLeftCorner(KHID, KHID);
    MatrixXd u_R_dense_inv = pinv_matrix(u_R_dense);
    u_R_inv_sparse = u_R_dense_inv.sparseView();
    SparseMatrix<double> u_Q_tmp = u* u_Q_R.colsPermutation();
    u_Q = u_Q_tmp * u_R_inv_sparse;
    u_Q.makeCompressed();
    u_Q.prune(EPS);
    
    //////
    MatrixXd L = (MatrixXd)u_Q.transpose() * A * u_Q;
    VectorXd L_diag = L.diagonal().cwiseAbs();
    /////////////////////////////////////
    // cout << "------------Symmetric Nystrom:"<< L_diag << endl;

    result.first = u_Q;
    result.first.makeCompressed();
    result.first.prune(EPS);
    
    result.second = L_diag.sparseView();
    result.second.prune(EPS);
    
    return result;
}
/////////////////////
/////////////////////
pair<SparseMatrix<double>,SparseMatrix<double> > pinv_symNystrom_sparse(SparseMatrix<double> X) // pseudoinverse of symmetric sparse matrix using nystrom method
{
    cout << "--------start of pinv_symNystrom_sparse----------"<< endl;
    int k = min(X.cols(),X.rows());
    k = min(k,2*KHID);
    SparseMatrix<double> result;
    /*// this route is too expensive
    pair<SparseMatrix<double>, SparseVector<double> > my_svd;
     my_svd = SVD_symNystrom_sparse(X, k);
    
    SparseMatrix<double>sig;
    sig.resize(k, k);
    for( int i = 0; i < k; i++)
    {
        sig.coeffRef(i, i)=pinv_num(my_svd.second.coeff(i, i));
    }
    sig.makeCompressed();
    sig.prune(EPS);
    
    result = my_svd.first * sig * my_svd.first.transpose();
    */
    // alternative route
    // C
    //cout << "----------start of random_embedding_mat-------"<<endl;
    SparseMatrix<double> random_mat = random_embedding_mat((long)X.cols(),k);
    //cout << "--------end of random embedding mat --------"<<endl;
    SparseMatrix<double> C=X*random_mat;
    // pseudo inverse of C
    //cout << "------start of pinv_matrix------"<<endl;
    SparseMatrix<double> C_inv = pinv_matrix(C);
    //cout << "-------end of pinv_matrix----------"<<endl;
    //
    C_inv.makeCompressed(); C_inv.prune(EPS);
    // W
    SparseMatrix<double> W_1 = random_mat.transpose() * X;
    SparseMatrix<double> W_2 = W_1*random_mat;
    MatrixXd W = (MatrixXd) W_2;
    SparseMatrix<double> W_sparse = W.sparseView();
    W_sparse.makeCompressed();W_sparse.prune(EPS);
    
    // pinv(X)
    // pinv(X) = pinv(C') * W pinv(C);
    //    SparseMatrix<double> result_1 = C_inv.transpose()*W_sparse;
    // result = result_1 * C_inv;

    //    result = C_inv.transpose() * W_sparse * C_inv;
    // result.makeCompressed();
    // result.prune(EPS);
    pair<SparseMatrix<double>, SparseMatrix<double> > result_new;
    result_new.first = C_inv;
    result_new.second = W_sparse;
    return result_new;
}
//////////////////////////
SparseMatrix<double> sqrt_symNystrom_sparse(SparseMatrix<double> X,int k) // computing the square root of a symmetric sparse matrix using nystrom method
{
    SparseMatrix<double> result;
    pair<SparseMatrix<double>, SparseVector<double> > my_svd;
    my_svd = SVD_symNystrom_sparse(X, k);
    
    SparseMatrix<double>sig;
    sig.resize(k, k);
    for( int i = 0; i < k; i++)
    {
      sig.coeffRef(i, i)=sqrt(fabs(my_svd.second.coeff(i, i)));
    }
    sig.makeCompressed();
    sig.prune(EPS);    
    result = my_svd.first * sig * my_svd.first.transpose();
    result.makeCompressed();
    result.prune(EPS);
    return result;
}

///////////////////////////////
pair<pair<SparseMatrix<double>,SparseMatrix<double> >, SparseVector<double> > SVD_asymNystrom_sparse(SparseMatrix<double> X,int k) // computing the sparse asymmetric k-svd using nystrom method
{
    pair<pair<SparseMatrix<double>,SparseMatrix<double> >, SparseVector<double> > result;
    pair<SparseMatrix<double>, SparseVector<double> > my_column;
    pair<SparseMatrix<double>, SparseVector<double> > my_row;
    SparseMatrix<double> A = X*X.transpose();
    SparseMatrix<double> B = X.transpose()*X;
    my_column = SVD_symNystrom_sparse (A, k);
    my_row    = SVD_symNystrom_sparse (B, k);
    
    SparseVector<double> sig1 = my_column.second;
    SparseVector<double> sig2 = my_row.second;
    SparseVector<double>sig;
    sig.resize(k);
    for( int i = 0; i < k; i++)
    {
      sig.coeffRef(i)=sqrt(sqrt(fabs(sig1.coeff(i)))*sqrt(fabs(sig2.coeff(i))));
    }
    sig.prune(EPS);
    
    result.first.first = my_column.first;
    result.first.first.makeCompressed();
    result.first.first.prune(EPS);
    
    result.first.second =my_row.first;
    result.first.second.makeCompressed();
    result.first.second.prune(EPS);
    
    result.second = sig;
    result.second.prune(EPS);
    cout <<"--------!!!!!!!!!!!!!!!!!!!!!!!!!!!Asymetric Nystrom Singular Values:"<< endl<<(VectorXd)sig<<endl;
    return result;
}
//////////////////////
pair<SparseMatrix<double>,SparseMatrix<double> > pinv_asymNystrom_sparse(SparseMatrix<double> X) // pseudoinverse in the sparse asymmetric case using nystrom method
{

    int k = min(X.cols(),X.rows());
    k = min(k,2*KHID);
    //    SparseMatrix<double> result;
    // one alternative pinv(X'*X) * X';
    pair<SparseMatrix<double>,SparseMatrix<double> > result;
    SparseMatrix<double> XX= X.transpose()*X;
    result = pinv_symNystrom_sparse(XX);
    //result_new_f = result*X.transpose();
    /*// one alternative
    pair<SparseMatrix<double>, SparseVector<double> > my_column;
    pair<SparseMatrix<double>, SparseVector<double> > my_row;
    SparseMatrix<double> A= X*X.transpose();
    SparseMatrix<double> B=X.transpose()*X;
    my_column = SVD_symNystrom_sparse(A, k);
    my_row =  SVD_symNystrom_sparse(B, k);
    
    SparseVector<double> sig1 = my_column.second;
    SparseVector<double> sig2 = my_row.second;
    SparseMatrix<double>sig;
    sig.resize(KHID, KHID);
    for( int i = 0; i < KHID; i++)
    {
      sig.coeffRef(i, i)=pinv_num(sqrt(sqrt(fabs(sig1.coeff(i)))*sqrt(fabs(sig2.coeff(i)))));
    }
    sig.makeCompressed();
    sig.prune(EPS);
    
    result = my_column.first * sig * my_row.first.transpose();
    */
    // result.makeCompressed();
    //result.prune(EPS);
    return result;
}

//////////////////////////////////////////////////////
SparseMatrix<double> sqrt_asymNystrom_sparse(SparseMatrix<double> X) // square root of an asymmetric sparse matrix using nystrom method
{
    int k = min(X.cols(),X.rows());
    k = min(k,KHID);
    SparseMatrix<double> result;
    pair<SparseMatrix<double>, SparseVector<double> > my_column;
    pair<SparseMatrix<double>, SparseVector<double> > my_row;
    SparseMatrix<double> A= X*X.transpose();
    SparseMatrix<double> B=X.transpose()*X;
    my_column = SVD_symNystrom_sparse(A, k);
    my_row =  SVD_symNystrom_sparse(B, k);
    
    SparseVector<double> sig1 = my_column.second;
    SparseVector<double> sig2 = my_row.second;
    SparseMatrix<double>sig;
    sig.resize(k, k);
    for( int i = 0; i < k; i++)
    {
      sig.coeffRef(i, i)=sqrt(sqrt(sqrt(fabs(sig1.coeff(i)))*sqrt(fabs(sig2.coeff(i)))));
    }
    sig.makeCompressed();
    sig.prune(EPS);
    
    result = my_column.first * sig * my_row.first.transpose();
    result.makeCompressed();
    result.prune(EPS);
    return result;
}

 
////////////////////////////////////////////////////
// pseudo inverse tested
double pinv_num (double pnum) // inverse of a scalar upto a tolerance
{
    double pnum_inv;
    if (fabs( pnum) > pinvtoler )
        pnum_inv=1.0/pnum;
    else pnum_inv=0;
    return pnum_inv;
}
/////////////////////////////////////////////////
Eigen::VectorXd pinv_vector (Eigen::VectorXd pinvvec) // elementwise inverse of a dense vector upto a tolerance
{
    Eigen::VectorXd singularValues_inv(pinvvec.size());
    for ( int i=0; i<pinvvec.size(); ++i) {
      if ( fabs(pinvvec(i)) > pinvtoler )
            singularValues_inv(i)=1.0/pinvvec(i);
        else singularValues_inv(i)=0;
    }
    return singularValues_inv;
}
Eigen::SparseVector<double> pinv_vector (Eigen::SparseVector<double> pinvvec) // elementwise inverse of a sparse vector upto a tolerance
{
    Eigen::SparseVector<double> singularValues_inv;
    singularValues_inv.resize(pinvvec.size());
    
    for ( int i=0; i<pinvvec.size(); ++i) {
      if (fabs( pinvvec.coeff(i)) > pinvtoler)
            singularValues_inv.coeffRef(i)=1.0/pinvvec.coeff(i);
        else singularValues_inv.coeffRef(i)=0;
    }
    singularValues_inv.prune(EPS);
    return singularValues_inv;
}
//////////////////////////////////////////////////
Eigen::MatrixXd pinv_matrix( Eigen::MatrixXd pinvmat) // pseudoinverse of a matrix computed using svd
{
    
    pair<pair<Eigen::MatrixXd,Eigen::MatrixXd>, Eigen::VectorXd> U_L =latenttree_svd (pinvmat);
    Eigen::VectorXd singularValues_inv=U_L.second;
    for ( long i=0; i<min(pinvmat.cols(),pinvmat.rows()); ++i) {
      if ( fabs(U_L.second(i) )> pinvtoler )
            singularValues_inv(i)=1.0/U_L.second(i);
        else singularValues_inv(i)=0;
    }
    pinvmat= (U_L.first.second*singularValues_inv.asDiagonal()*U_L.first.first.transpose());
    return pinvmat;
}




Eigen::SparseMatrix<double> pinv_matrix( Eigen::SparseMatrix<double> A) // pseudoinverse of a sparse matrix using the interface between svdlibc and eigen
{
  //cout << "---pinv_matrix:dimension: "<< A.rows()<<" , "<< A.cols()<<endl;
  //cout << "---pinv_matrix:nonZeros():"<<A.nonZeros()<<endl;
   //incorporate sparseSVD
   EigenSparseSVD u_l = sparse_svd(A,KHID);
   Eigen::MatrixXd u = u_l.left_singular_vectors.leftCols(KHID);
   Eigen::MatrixXd v = u_l.right_singular_vectors.leftCols(KHID);
   Eigen::VectorXd s = u_l.singular_values.head(KHID);
   //////
   Eigen::VectorXd singularValues_inv(s.size());
   singularValues_inv = s;
   for ( long i=0; i<KHID; ++i) {
     if ( fabs(s(i)) > pinvtoler )
           singularValues_inv(i)=1.0/s(i);
       else singularValues_inv(i)=0;
   }
   Eigen::SparseMatrix<double> pinvmat;
   pinvmat.resize(A.cols(),A.rows());
   pinvmat = (v * singularValues_inv.asDiagonal()* u.transpose()).sparseView();
   pinvmat.makeCompressed();
   pinvmat.prune(EPS);
   return pinvmat;

}


///////////////////////////////////////////////////
Eigen::MatrixXd sqrt_matrix(Eigen::MatrixXd pinvmat) // square root of a matrix using svd
{
    Eigen::MatrixXd sqrtmat;
    pair<pair<Eigen::MatrixXd,Eigen::MatrixXd>, Eigen::VectorXd> U_L =latenttree_svd (pinvmat);
    Eigen::VectorXd singularValues_sqrt=U_L.second.head(KHID);
    Eigen::MatrixXd left_sing_vec = U_L.first.first.leftCols(KHID);
    Eigen::MatrixXd right_sing_vec = U_L.first.second.leftCols(KHID);
    for ( long i=0; i<KHID; ++i) {
      singularValues_sqrt(i)=sqrt(fabs(U_L.second(i)));
    }
    
    sqrtmat= (left_sing_vec*singularValues_sqrt.asDiagonal()*right_sing_vec.transpose());
    
    ////////
    return sqrtmat;
}

////////////////////////////////////////////////////////
void second_whiten(SparseMatrix<double> Gx_a, SparseMatrix<double> Gx_b, SparseMatrix<double> Gx_c, SparseMatrix<double> &W, SparseMatrix<double> &Z_B, SparseMatrix<double> &Z_C, VectorXd &mu_a, VectorXd &mu_b, VectorXd &mu_c) // second whitening method for the detecting overlapping communities - for details, refer the appendix of the arxiv version
{
  double nx = (double)Gx_a.rows();
    SparseVector<double> my_ones_a = (VectorXd::Ones(Gx_a.rows())).sparseView();
    SparseVector<double> my_ones_b = (VectorXd::Ones(Gx_b.rows())).sparseView();
    SparseVector<double> my_ones_c = (VectorXd::Ones(Gx_c.rows())).sparseView();
    SparseVector<double> mu_a_sparse = my_ones_a.transpose() * Gx_a;
    SparseVector<double> mu_b_sparse = my_ones_b.transpose() * Gx_b;
    SparseVector<double> mu_c_sparse = my_ones_c.transpose() * Gx_c;
    mu_a_sparse = mu_a_sparse/((double)nx);
    mu_b_sparse = mu_b_sparse/((double)nx);
    mu_c_sparse = mu_c_sparse/((double)nx);
    mu_a = ((VectorXd)mu_a_sparse);
    mu_b = ((VectorXd)mu_b_sparse);
    mu_c = ((VectorXd)mu_c_sparse);
    double inv_nx = 1/((double)nx);
    SparseMatrix<double> Z_B_numerator = Gx_a.transpose() * Gx_c;// NA * NC
    Z_B_numerator = inv_nx * Z_B_numerator;       // NA * NC
    SparseMatrix<double> Z_B_denominator = Gx_b.transpose()*Gx_c;// NB * NC

    Z_B_denominator =inv_nx * Z_B_denominator;//NB * NC
    //cout << "Z_B_denominator.nonZeros():" <<Z_B_denominator.nonZeros()<<endl; 
    
    SparseMatrix<double> Z_C_numerator = Gx_a.transpose() * Gx_b; // NA * NB
    Z_C_numerator = inv_nx *  Z_C_numerator;       // NA * NB??????????????????
    SparseMatrix<double> Z_C_denominator = Gx_c.transpose()*Gx_b;//NC * NB
    Z_C_denominator = inv_nx * Z_C_denominator;//NC * NB??????????????
    //cout << "Z_C_denominator.nonZeros():" <<Z_C_denominator.nonZeros()<< endl;
    cout << "--------------starting to calculate Z_B implicitly---------------------------"<< endl;
    // if dimensions are too large, use symmetric approximation and requires a even partition nA=nB=nC
    pair<SparseMatrix<double>,SparseMatrix<double> > pairB;
    pair<SparseMatrix<double>,SparseMatrix<double> > pairC;
    if(NA+NB+NC<200000)
    {
        pairB= pinv_asymNystrom_sparse(Z_B_denominator);
    }
    else
    {
        if (NA!=NB or NA!=NC or NB!=NC)
        {
            printf("Error! For large datasets, we need even partition (|A|=|B|=|C|).\n"); fflush(stdout);
            exit(1);
        }
        else
        {
            // symmetric approximation
            pairB= pinv_symNystrom_sparse(Z_B_denominator);
        }
    }
    //
    //pair<SparseMatrix<double>, SparseMatrix<double> > pairB= pinv_symNystrom_sparse(Z_B_denominator);
    SparseMatrix<double> C_inv_pairB =pairB.first;
    SparseMatrix<double> W_pairB = pairB.second;
    
    cout << "--------------starting to calculate Z_C implicitly---------------------------"<< endl;
    // if dimensions are too large, use symmetric approximation and requires a even partition nA=nB=nC
    if(NA+NB+NC<200000)
    {
         pairC= pinv_asymNystrom_sparse(Z_C_denominator);
    }
    else
    {
        if (NA!=NB or NA!=NC or NB!=NC)
        {
            printf("Error! For large datasets, we need even partition (|A|=|B|=|C|).\n"); fflush(stdout);
            exit(1);
        }
        else
        {
            // symmetric approximation
             pairC= pinv_symNystrom_sparse(Z_C_denominator);
        }
    }
    //
    //pair<SparseMatrix<double>, SparseMatrix<double> > pairC= pinv_symNystrom_sparse(Z_C_denominator);
    SparseMatrix<double> C_inv_pairC =  pairC.first;
    SparseMatrix<double> W_pairC = pairC.second;
    
#ifdef CalErrALL
    SparseMatrix<double> Z_B_tmp1 = Z_B_denominator.transpose();
    SparseMatrix<double> Z_C_tmp1 = Z_C_denominator.transpose();
    SparseMatrix<double> Z_B_tmp2;
    SparseMatrix<double> Z_C_tmp2;
    if(NA+NB+NC<200000)
    {
        Z_B_tmp2 = C_inv_pairB * Z_B_tmp1;
        Z_C_tmp2 = C_inv_pairC * Z_C_tmp1;
    }
    else
    {
        if (NA!=NB or NA!=NC or NB!=NC)
        {
            printf("Error! For large datasets, we need even partition (|A|=|B|=|C|).\n"); fflush(stdout);
            exit(1);
        }
        else
        {
            // symmetric approximation
            Z_B_tmp2 = Z_B_tmp1;
            Z_C_tmp2 = Z_C_tmp1;
        }
    }
    SparseMatrix<double> Z_B_tmp3 = W_pairB * Z_B_tmp2;
    SparseMatrix<double> Z_C_tmp3 = W_pairC * Z_C_tmp2;
    SparseMatrix<double> Z_B_tmp4 = C_inv_pairB.transpose() * Z_B_tmp3;
    SparseMatrix<double> Z_C_tmp4 = C_inv_pairC.transpose() * Z_C_tmp3;
    SparseMatrix<double> Z_B = Z_B_numerator * Z_B_tmp4;
    SparseMatrix<double> Z_C = Z_C_numerator * Z_C_tmp4;
#endif

    
    // compute M2_alpha0
    Gx_c.prune(EPS); Gx_b.prune(EPS);
    cout << "-----------calculating M2---------------"<<endl;
    // whiten = pinv(C)' * W^(1/2)';
    // C
    //cout << "--------------generating random_mat for M2---------------"<<endl;
    SparseMatrix<double> random_mat = random_embedding_mat((long) Gx_a.cols(),KHID);
    SparseMatrix<double> M2_tmp1 = Z_B_numerator.transpose() * random_mat;
   
    SparseMatrix<double> M2_tmp2 = C_inv_pairB * M2_tmp1;
    SparseMatrix<double> M2_tmp3 = W_pairB.transpose() * M2_tmp2;
    SparseMatrix<double> M2_tmp4 = C_inv_pairB.transpose() * M2_tmp3;
    // if dimensions are too large, use symmetric approximation and requires a even partition nA=nB=nC
    SparseMatrix<double> M2_tmp5;
    if(NA+NB+NC<200000)
    {
         M2_tmp5 = Z_B_denominator * M2_tmp4;
    }
    else
    {
        if (NA!=NB or NA!=NC or NB!=NC)
        {
            printf("Error! For large datasets, we need even partition (|A|=|B|=|C|).\n"); fflush(stdout);
            exit(1);
        }
        else
        {
            // symmetric approximation
             M2_tmp5 = M2_tmp4;
        }
    }
    //
    
    
    
        SparseMatrix<double> M2_tmp6 = Z_C_denominator  *M2_tmp5;
    // if dimensions are too large, use symmetric approximation and requires a even partition nA=nB=nC
    SparseMatrix<double> M2_tmp7;
    if(NA+NB+NC<200000)
    {
         M2_tmp7 = Z_C_denominator.transpose() * M2_tmp6;
    }
    else
    {
        if (NA!=NB or NA!=NC or NB!=NC)
        {
            printf("Error! For large datasets, we need even partition (|A|=|B|=|C|).\n"); fflush(stdout);
            exit(1);
        }
        else
        {
            M2_tmp7 = M2_tmp6;
        }
    }
    
    SparseMatrix<double> M2_tmp8 = C_inv_pairC  *M2_tmp7;
    SparseMatrix<double> M2_tmp9 = W_pairC * M2_tmp8;
    SparseMatrix<double> M2_tmp10 = C_inv_pairC.transpose() * M2_tmp9;
    SparseMatrix<double> M2 =Z_C_numerator * M2_tmp10;
    M2.makeCompressed();
    M2.prune(EPS);
//    M2 = inv_nx * M2;
    cout << "----------M2.nonZeros():"<< M2.nonZeros() << endl;
    cout << "----------end of caluclating M2, start M2_alpha0-----"<<endl;
    // M2 -> M2_alpha0
     cout <<"-------------------computing square_mu_a_sparse--------"<<endl;
    double para = ((double)alpha0/((double)alpha0+1));
    SparseMatrix<double> tmp_square(M2.rows(),M2.rows());
    tmp_square.makeCompressed();
    for (int i = 0; i < M2.rows(); i++)
      {
	tmp_square.coeffRef(i,i)=mu_a(i)*mu_a(i);
      }
    SparseMatrix<double> tmp_square_proj = tmp_square * random_mat;
    tmp_square_proj.prune(EPS);
    
   
    mu_a_sparse.prune(EPS);
    SparseVector<double> mu_a_sparse_proj = mu_a_sparse.transpose()*random_mat;
    SparseMatrix<double> square_mu_a_sparse_proj = mu_a_sparse* mu_a_sparse_proj.transpose();
    square_mu_a_sparse_proj = square_mu_a_sparse_proj - tmp_square_proj;

    square_mu_a_sparse_proj = square_mu_a_sparse_proj*para;
    square_mu_a_sparse_proj.makeCompressed();
    square_mu_a_sparse_proj.prune(EPS);
    
    SparseMatrix<double> C =M2 - square_mu_a_sparse_proj;
     C.makeCompressed(); C.prune(EPS);
     cout << "-----------M2_alpha0:nonZeros()" << M2.nonZeros()<< "-------------"<<endl;
    // pseudo inverse of C
    cout <<"-----------C.nonZeros():"<< C.nonZeros()<< endl;
    cout <<"-----------C.first row:\n" << C.row(0)<< endl;
    SparseMatrix<double> C_inv = pinv_matrix(C);
    // W_nystrom
    MatrixXd W_nystrom = (MatrixXd)random_mat.transpose() * C;
    MatrixXd W_nystrom_sqrt = sqrt_matrix(W_nystrom);
    SparseMatrix<double> W_nystrom_sqrt_sparse = W_nystrom_sqrt.sparseView();
    W_nystrom_sqrt_sparse.makeCompressed();W_nystrom_sqrt_sparse.prune(EPS);
    // getting the witening matrix W
    W.resize(Gx_a.cols(),KHID);
    W = C_inv.transpose()*W_nystrom_sqrt_sparse.transpose();
    W.makeCompressed(); W.prune(EPS); 
    cout << "---------------------dimension of W : " << W.rows() <<" , " << W.cols() << "----------------"<< endl;
    cout << "-----------End of Whitening----------nonZeros() of W : " << W.nonZeros()<< endl;

}



void tensorDecom_alpha0(SparseMatrix<double> D_a_mat, VectorXd D_a_mu, SparseMatrix<double> D_b_mat, VectorXd D_b_mu, SparseMatrix<double> D_c_mat, VectorXd D_c_mu, VectorXd &lambda, MatrixXd & phi_new) // implementation of stochastic updates for the k \times k \times k tensor implicitly
{
    double error;
    
    MatrixXd A_random(MatrixXd::Random(KHID,KHID));
    MatrixXd phi_old;

    A_random.setRandom();
    HouseholderQR<MatrixXd> qr(A_random);
    phi_new = qr.householderQ();
    A_random.resize(0,0);
    
    long iteration = 0;
    double beta =(double) LEARNRATE;
    while (true)
    {
        long iii = iteration % NX;
        VectorXd D_a_g = D_a_mat.col((int)iii);
        VectorXd D_b_g = D_b_mat.col((int)iii);
        VectorXd D_c_g = D_c_mat.col((int)iii);
	if(iteration%20 == 0)        
        phi_old=phi_new;
        phi_new =Diff_Loss(D_a_g,D_b_g,D_c_g,D_a_mu,D_b_mu,D_c_mu,phi_old,beta);
///////////////////////////////////////////////
        if (iteration < MINITER)
        {}
        else
        {
            error = (phi_new - phi_old).norm();
            if (error < TOLERANCE or iteration > MAXITER )
            {
                
                break;
                
            }
        }
        
        iteration++;
    }
    Eigen::MatrixXd tmp_lambda = (((phi_new.array().pow(2)).colwise().sum()).pow(3.0/2.0)).transpose();
    lambda = tmp_lambda;
    lambda.normalize();
    phi_new = normc(phi_new);
}

MatrixXd Diff_Loss(VectorXd Data_a_g, VectorXd Data_b_g,VectorXd Data_c_g,VectorXd Data_a_mu,VectorXd Data_b_mu,VectorXd Data_c_mu, Eigen::MatrixXd phi,double beta) // computing the diff loss for whitening
{
    MatrixXd New_Phi;
    
    MatrixXd myvector=MatrixXd::Zero(KHID,KHID);
    
    for (int index_k = 0; index_k < KHID; index_k++)
    {
        MatrixXd one_eigenvec(KHID,1);
        one_eigenvec = phi.col(index_k);
        MatrixXd tmp = (one_eigenvec.transpose()*phi).array().pow(2);
        VectorXd tmp2 = tmp.row(0);
        MatrixXd tmp_mat = tmp2.asDiagonal();
        
        tmp_mat = phi * tmp_mat;
        VectorXd vector_term1 = 3*tmp_mat.rowwise().sum();
        VectorXd vector_term2 =-3*The_second_term(Data_a_g,Data_b_g,Data_c_g,Data_a_mu,Data_b_mu,Data_c_mu,one_eigenvec);
                                       
        myvector.col(index_k)=vector_term1+vector_term2;
    }
     New_Phi = phi - myvector*beta;
    
    return New_Phi;
}

VectorXd The_second_term(VectorXd Data_a_g,VectorXd Data_b_g,VectorXd Data_c_g,VectorXd Data_a_mu,VectorXd Data_b_mu,VectorXd Data_c_mu,VectorXd phi) // second part of the stochastic updates
{
    // phi is a VectorXd
  double para1 =((double) 2*alpha0*alpha0) / ((double)(alpha0+1)*(alpha0+2));
    double para2 =-alpha0/(alpha0+2);
    
    VectorXd Term1 = (phi.dot(Data_a_g))*(phi.dot(Data_b_g)) * Data_c_g;
    VectorXd Term2 = para1*(phi.dot(Data_a_mu))*(phi.dot(Data_b_mu))*Data_c_mu;
    VectorXd Term31 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_g))*Data_c_mu;
    VectorXd Term32 = para2*(phi.dot(Data_a_g))*(phi.dot(Data_b_mu))*Data_c_g;
    VectorXd Term33 = para2*(phi.dot(Data_a_mu))*(phi.dot(Data_b_g))*Data_c_g;
    VectorXd output =Term1+Term2+Term31+Term32+Term33;
    
    return output;
}
// THE END





///////////////////////////////////////////////////////////////////////////////////

////////////
Eigen::MatrixXd condi2condi(Eigen::MatrixXd p_x_h, Eigen::VectorXd p_h) // compute the posterior
{
    Eigen::MatrixXd p_x_h_joint = Condi2Joint(p_x_h, p_h); // child*parent, parent.
    Eigen::VectorXd p_x = p_x_h_joint.rowwise().sum();
    // Eigen::VectorXd p_h = p_x_h_joint.colwise().sum();
    Eigen::MatrixXd p_h_x_joint = p_x_h_joint.transpose();
    Eigen::MatrixXd p_h_x = joint2conditional(p_h_x_joint, p_x);
    return p_h_x;
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// Parameter Estimation
Eigen::MatrixXd concatenation_matrix (Eigen::MatrixXd A, Eigen::MatrixXd B) // concatenate 2 eigen matrices
{
    Eigen::MatrixXd C(A.rows()+B.rows(),A.cols());
    //C.resize(A.rows()+B.rows(),A.cols());
    C<<A,B;
    return C;
}
///////////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd concatenation_vector (Eigen::VectorXd A, Eigen::VectorXd B) // concatenate 2 eigen vectors
{
    Eigen::VectorXd C(A.size()+B.size());
    // C.resize(A.size()+B.size());
    C<<A,B;
    return C;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////This is for Parameter estimation
///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

// normc tested
Eigen::MatrixXd normc(Eigen::MatrixXd phi) // normalize the columns of a matrix
{
    for(int i = 0; i < phi.cols() ; i++)
    {
        phi.col(i).normalize();
    }    
    return phi;
}

//////////////////////
Eigen::MatrixXd normProbMatrix(Eigen::MatrixXd P)
{
    // each column is a probability simplex
    Eigen::MatrixXd P_norm(P.rows(),P.cols());
    int nCol =(int) P.cols();
    for (int col = 0; col < nCol ; col++)
    {
        Eigen::VectorXd P_vec = P.col(col);
        if (P_vec == Eigen::VectorXd::Zero(P.rows()))
        {
            P_norm.col(col)=P_vec;
        }
        else
        {
            //Eigen::VectorXd tmp(P_vec.size());
            double P_positive;// = P_vec(P_vec >0 | P_vec ==0).sum();
            double P_negative;// = - P_vec(P_vec <0).sum();
            
            for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
            {
                if (P_vec(row_idx)>=0)
                {
                    P_positive = P_positive + P_vec(row_idx);
                }
                else
                {
                    P_negative = P_negative + P_vec(row_idx);
                }
            }
            if (P_positive > P_negative or P_positive == P_negative)
            {
                for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
                {
                    if (P_vec(row_idx)<0)
                    {
                        P_vec(row_idx)=EPS;
                    }
                    
                }
            }
            else
            {
                P_vec=-P_vec;
                for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
                {
                    if (P_vec(row_idx)<0)
                    {
                        P_vec(row_idx)=EPS;
                    }
                    
                }
            }
            P_vec = P_vec /P_vec.sum();
            P_norm.col(col)=P_vec;
        }
        
    }
    return P_norm;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::VectorXd normProbVector(Eigen::VectorXd P)
{
    Eigen::VectorXd P_norm;
    Eigen::VectorXd P_vec = P;
    if (P_vec == Eigen::VectorXd::Zero(P.size()))
    {
        P_norm=P_vec;
    }
    else
    {
        //Eigen::VectorXd tmp(P_vec.size());
        double P_positive;// = P_vec(P_vec >0 | P_vec ==0).sum();
        double P_negative;// = - P_vec(P_vec <0).sum();
        
        for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
        {
            if (P_vec(row_idx)>=0)
            {
                P_positive = P_positive + P_vec(row_idx);
            }
            else
            {
                P_negative = P_negative + P_vec(row_idx);
            }
        }
        if (P_positive > P_negative or P_positive == P_negative)
        {
            for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
            {
                if (P_vec(row_idx)<0)
                {
                    P_vec(row_idx)=EPS;
                }
                
            }
        }
        else
        {
            P_vec=-P_vec;
            for (int row_idx = 0; row_idx < P_vec.size(); row_idx++)
            {
                if (P_vec(row_idx)<0)
                {
                    P_vec(row_idx)=EPS;
                }
                
            }
        }
        P_vec = P_vec /P_vec.sum();
        P_norm=P_vec;
    }
    return P_norm;
}
//////////////////////////////

//////////////
Eigen::MatrixXd Condi2Joint(Eigen::MatrixXd Condi, Eigen::VectorXd Pa) // conditional to joint
{
    Eigen::MatrixXd Joint(Condi.rows(), Condi.cols());
    for (int cols = 0; cols < Condi.cols(); cols++)
    {
        Joint.col(cols) = Condi.col(cols)*Pa(cols);
    }
    cout << Joint << endl;
    return Joint;
    
}

//////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/////////////////
Eigen::MatrixXd joint2conditional(Eigen::MatrixXd edgePot,Eigen::VectorXd pa)// pa is the second dimension
{
    Eigen::MatrixXd Conditional(edgePot.rows(),edgePot.cols());
    for(int col = 0; col < edgePot.cols(); col++)
    {
        //Conditional.col(col) = edgePot.col(col)/(pa(col));
        if (pa(col)> EPS)
        {
            Conditional.col(col) = edgePot.col(col)/(pa(col));
        }
        else
        {
            Conditional.col(col)=Eigen::VectorXd::Zero(edgePot.rows());
        }
    }
    
    return Conditional;
}
////////////
//////////
///////////////////////////////////////////
// copy and paste the following function to use it for reading
// sparse matrix binary and weighted case for reading G_XA, G_XB and G_XC
Eigen::SparseMatrix<double> read_G_sparse(char *file_name, char *G_name,int N1, int N2) // input: file name, adjacent matrix name, NA/NB/NC, output: sparse matrix
{
    printf("reading %s\n", G_name); fflush(stdout);
    Eigen::SparseMatrix<double> G_mat(N1, N2); // NX \times (NA or NB or NC)
    G_mat.makeCompressed();
    double r_idx, c_idx; // row and column indices - matlab style
#ifdef EXPECTED
    double val;
#endif
    FILE* file_ptr = fopen(file_name, "r"); // opening G_name
    if(file_ptr == NULL) // exception handling if reading G_name fails
    {
        printf("reading adjacency submatrix failed\n"); fflush(stdout);
        exit(1);
    }
    while(!feof(file_ptr)) // reading G_name
    {
        fscanf(file_ptr, "%lf", &r_idx); // first read in row then col
        fscanf(file_ptr, "%lf", &c_idx);
# ifdef EXPECTED
        fscanf(file_ptr, "%lf", &val);
        G_mat.coeffRef(r_idx-1, c_idx-1) = val; // this is now modified in r and c idx; reads in weighted also;
# endif
# ifdef BINARY
        G_mat.coeffRef(r_idx-1, c_idx-1) = 1;
# endif
    }
    fclose(file_ptr);
    return G_mat;
}
//////////////////////
EigenSparseSVD sparse_svd(Eigen::SparseMatrix<double> eigen_sparse_matrix, int rank) // input arguments are a sparse matrix in csc format in eigen toolkit and the rank parameter which corresponds to the number of singular values required; note that this is in double precision
{
    int i, j; // loop variables
    EigenSparseSVD eigen_svd_variable; // to be returned
    SMat svdlibc_sparse_matrix = svdNewSMat(eigen_sparse_matrix.rows(), eigen_sparse_matrix.cols(), eigen_sparse_matrix.nonZeros()); // allocate dynamic memory for a svdlibc sparse matrix
    if(svdlibc_sparse_matrix == NULL)
    {
        printf("memory allocation for svdlibc_sparse_matrix variable in the sparse_svd() function failed\n");
        fflush(stdout);
        exit(3);
    }
    SVDRec svd_result = svdNewSVDRec(); // allocate dynamic memory for a svdlibc svd record for storing the result of applying the lanczos method on the input matrix
    if(svd_result == NULL)
    {
        printf("memory allocation for svd_result variable in the sparse_svd() function failed\n");
        fflush(stdout);
        exit(3);
    }
    int iterations = 0; // number of lanczos iterations - 0 means until convergence
    double las2end[2] = {-1.0e-30, 1.0e-30}; // tolerance interval
    double kappa = 1e-6; // another tolerance parameter
    double copy_tol = 1e-6; // tolerance threshold for copying from svdlibc to eigen format
    
    eigen_sparse_matrix.makeCompressed(); // very crucial - this ensures correct formatting with correct inner and outer indices
    if(rank == 0) // checking for full rank svd option
        rank = ( (eigen_sparse_matrix.rows()<eigen_sparse_matrix.cols())?eigen_sparse_matrix.rows():eigen_sparse_matrix.cols() );
    
    i = 0;
    while(i < eigen_sparse_matrix.nonZeros()) // loop to assign the non-zero values
    {
        svdlibc_sparse_matrix->value[i] = *(eigen_sparse_matrix.valuePtr()+i);
        i++;
    }
    i = 0;
    while(i < eigen_sparse_matrix.nonZeros()) // loop to assign the inner indices
    {
        svdlibc_sparse_matrix->rowind[i] = *(eigen_sparse_matrix.innerIndexPtr()+i);
        i++;
    }
    i = 0;
    while(i < eigen_sparse_matrix.cols()) // loop to assign the outer indices
    {
        svdlibc_sparse_matrix->pointr[i+1] = *(eigen_sparse_matrix.outerIndexPtr()+i+1); // both must be +1 - this has been tested; refer comment in the struct in svdlib.h to verify
        i++;
    }   
    svd_result = svdLAS2(svdlibc_sparse_matrix, rank, iterations, las2end, kappa); // computing the sparse svd of the input matrix using lanczos method
    for(i = 0; i < rank; i++) // update the rank based on zero singular value check; used for adaptively resizing the resultant
        if(svd_result->S[i] == 0)
        {
            rank = i;
            break;
        }
    eigen_svd_variable.singular_values.resize(rank); // allocating memory for our svd struct variable members
    eigen_svd_variable.left_singular_vectors.resize(eigen_sparse_matrix.rows(), rank); // allocating memory for our svd struct variable members
    eigen_svd_variable.right_singular_vectors.resize(eigen_sparse_matrix.cols(), rank); // allocating memory for our svd struct variable members
    
    // note that efficiency can be increased by avoiding this copy from svdlib to eigen format and using blas-style function calling; but this was done to facilitate subsequent operations in other parts of the algorithm
    for(i = 0; i < rank; i++) // loop to copy the singular values
        eigen_svd_variable.singular_values(i) = svd_result->S[i];
    for(i = 0; i < rank; i++) // loop to copy the left singular vectors
        for(j = 0; j < eigen_sparse_matrix.rows(); j++)
            if(fabs(svd_result->Ut->value[i][j]) > copy_tol) // checking numerical tolerance
                eigen_svd_variable.left_singular_vectors(j, i) = svd_result->Ut->value[i][j]; // transpose while copying
            else
                eigen_svd_variable.left_singular_vectors(j, i) = 0; // transpose while copying
    
    for(i = 0; i < rank; i++) // loop to copy the right singular vectors
        for(j = 0; j < eigen_sparse_matrix.cols(); j++)
            if(fabs(svd_result->Vt->value[i][j]) > copy_tol) // checking numerical tolerance
                eigen_svd_variable.right_singular_vectors(j, i) = svd_result->Vt->value[i][j]; // transpose while copying
            else
                eigen_svd_variable.right_singular_vectors(j, i) = 0; // transpose while copying
    
    svdFreeSVDRec(svd_result); // free the dynamic memory allocated for the svdlibc svd record
    svdFreeSMat(svdlibc_sparse_matrix); // free the dynamic memory allocated for the svdlibc sparse matrix
    
    return eigen_svd_variable;
}
/////////////
