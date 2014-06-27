//
//  IOfn.cpp
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#pragma once
#include "stdafx.h"
using namespace Eigen;
using namespace std;
extern int DATATYPE; // ==1 is the expected, ==0 is the binary. 


///////////////////////////////////////////
// copy and paste the following function to use it for reading
// sparse matrix binary and weighted case for reading G_XA, G_XB and G_XC
Eigen::SparseMatrix<double> read_G_sparse(char *file_name, char *G_name, int N1, int N2) // input: file name, adjacent matrix name, NA/NB/NC, output: sparse matrix
{
	printf("reading %s\n", G_name); fflush(stdout);
	Eigen::SparseMatrix<double> G_mat(N1, N2); // NX \times (NA or NB or NC)
	G_mat.makeCompressed();
	double r_idx, c_idx; // row and column indices - matlab style
	double val;

	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading adjacency submatrix failed\n"); fflush(stdout);
		exit(1);
	}
	while (!feof(file_ptr)) // reading G_name
	{
		fscanf(file_ptr, "%lf", &r_idx); // first read in row then col
		fscanf(file_ptr, "%lf", &c_idx);
		if (DATATYPE == 1)
		{
			fscanf(file_ptr, "%lf", &val);
			G_mat.coeffRef(r_idx - 1, c_idx - 1) = val; // this is now modified in r and c idx; reads in weighted also;
		}
		else
		G_mat.coeffRef(r_idx - 1, c_idx - 1) = 1;
	}
	fclose(file_ptr);
	return G_mat;
}


int write_pi(char *filename, SparseMatrix<double> mat)
{
	fstream f(filename, ios::out);
	for (long k = 0; k<mat.outerSize(); ++k)
	for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
	{
		f << it.row() + 1 << "\t" << it.col() + 1 << "\t" << it.value() << endl;
	}
	f.close();
	return 0;
}


void furongprintVector(double value[], long len, char *character) // print the elements of an array
{
	for (long i = 0; i<len; i++)
	{
		printf("%s=%.10f\n", character, value[i]);
	}
}