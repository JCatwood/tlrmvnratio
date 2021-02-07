// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <R_ext/Lapack.h>
#include <R_ext/Print.h>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mvphi.h"
#include "morton.h"
#include "mvnkernel.h"
#include "misc.h"
#include "aca.h"
#include "cholesky.h"
#include "covariance.h"
#include "mvn.h"
#include "uncompress.h"
#include "tlr.h"
#include "tlrmvn.h"
#include "tlr_tlr_add_qr.h"
#include "uni_reorder.h"
#include "blk_reorder.h"
#include "recur_blk_reorder.h"
#include "mvt.h"
#include "tlrmvt.h"
#include "cov_kernel.h"

using namespace std;
using namespace Eigen;

typedef std::chrono::time_point<std::chrono::steady_clock> TimeStamp;


// [[Rcpp::export]]
Rcpp::List tlrmvnratio_internal(Eigen::VectorXd a, Eigen::VectorXd b, Eigen::MatrixXd 
	covM, int m, double epsl, int N)
{
	// tmp var
	int n = covM.rows() - 1;
	int ns = 10;
	int fail;
	double v, e;
	int scaler;
	TimeStamp start, end;
	double timeCovM, timeChol, timeInt;
	// scale a, b, covM
	{
		VectorXd diagVec = covM.diagonal();
		diagVec.noalias() = diagVec.unaryExpr([](double x){return 
			1.0/sqrt(x);});
		auto diagM = diagVec.asDiagonal();
		a.noalias() = diagM * a;
		b.noalias() = diagM * b;
		covM.noalias() = diagM * covM;
		covM.noalias() = covM * diagM;
	}
	// build tlr covM, extend a and b if necessary, resize covM after
	start = std::chrono::steady_clock::now();
	vector<MatrixXd> B;
	vector<TLRNode> UV;
	int allocSz = max(m >> 2, 20);
	double epslACA = epsl / (double) m;
	tlr_aca_covM(covM.block(0, 0, n, n), B, UV, m, epslACA, allocSz);
	int lastBlkDim = n % m;
	if(lastBlkDim > 0)
		Rcpp::stop("Block size should be a factor of n\n");
	RowVectorXd lastRow = covM.row(n);
	covM.resize(0, 0);
	end = std::chrono::steady_clock::now();
	timeCovM = std::chrono::duration<double>(end - start).count();
	// mem alloc
	int lworkDbl = max(19*m*m+16*m+4*n, (5*n + 4*m + 25)*N + 2 * ns + m);
	int lworkInt = 4*N + 2*n + 2*m + ns + 1;
	double *workDbl = new double[lworkDbl];
	int *workInt = new int[lworkInt];
	// recur blk reorder
	start = std::chrono::steady_clock::now();
	double *y = workDbl;
	double *acp = y + n;
	double *bcp = acp + n;
	double *p = bcp + n;
	double *subworkDbl = p + n;
	int lsubworkDbl = 19*m*m + 16*m;
	if(subworkDbl + lsubworkDbl > workDbl + lworkDbl)
		Rcpp::stop("Memory overflow\n");
	int *idx = workInt;
	int *subworkInt = idx + n;
	int lsubworkInt = max(2*m, n);
	if(subworkInt + lsubworkInt > workInt + lworkInt)
		Rcpp::stop("Memory overflow\n");
	copy(a.data(), a.data()+n, acp);
	copy(b.data(), b.data()+n, bcp);
	iota(idx, idx+n, 0);
	fail = recur_blk_reorder(B, UV, lastRow, acp, bcp, p, y, idx, epsl, subworkDbl, 
		lsubworkDbl, subworkInt, lsubworkInt);
	if(fail)
		Rcpp::stop("TLR Cholesky failed. Either the original covariance "
			"matrix is not positive definite or the error in Cholesky "
			"factorization is significant\n");
	reorder(a.data(), idx, n, subworkInt, lsubworkInt);
	reorder(b.data(), idx, n, subworkInt, lsubworkInt);
	end = std::chrono::steady_clock::now();
	timeChol = std::chrono::duration<double>(end - start).count();
	// call the tlrmvn function
	start = std::chrono::steady_clock::now();
	tlrmvn(N, B, UV, lastRow, a, b, v, e, ns, scaler, workDbl, lworkDbl, 
		workInt, lworkInt);
	end = std::chrono::steady_clock::now();
	timeInt = std::chrono::duration<double>(end - start).count();
	// compute average rank
	int rkSum = 0;
	for(auto &tile : UV) rkSum += tile.crtColNum;
	int rkAvg = rkSum / UV.size();
	// mem release
	delete[] workDbl;
	delete[] workInt;
	// return a list
	return Rcpp::List::create(Rcpp::Named("Estimation") = v,
		Rcpp::Named("Building TLR covariance matrix time") = timeCovM,
		Rcpp::Named("Recursive block reordering time") = timeChol,
		Rcpp::Named("Monte Carlo time") = timeInt,
		Rcpp::Named("Average rank") = rkAvg);
}


