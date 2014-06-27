//
//  IOfn.h
//  latenttree
/*******************************************************
* Copyright (C) 2014 {Furong Huang} <{furongh@uci.edu}>
*
* This file is part of {community detection project}.
*
* All rights reserved.
*******************************************************/
#ifndef __CommunityDetection__IOfn__
#define __CommunityDetection__IOfn__
#include "stdafx.h"

using namespace Eigen;
using namespace std;

SparseMatrix<double> read_G_sparse(char *file_name, char *G_name, int N1, int N2);

int write_pi(char *filename, SparseMatrix<double> mat);
void furongprintVector(double value[], long len, char *character);
#endif