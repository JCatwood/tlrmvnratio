#include <RcppEigen.h>
#include <R_ext/Print.h>
#include <R_ext/Lapack.h>
#include <R.h>
#include <algorithm>
#include <ctime>
#include "mvnkernel.h"
#include "tlr.h"
#include "misc.h"
#include "mvphi.h"

#define MIN_PROB_EXPO -1E3

using namespace std;
using namespace Eigen;

/*
        lworkDbl should be no smaller than (5n+4m+25)N + 2ns + m
        lworkInt should be no smaller than 4N + n + ns + 1
        2019/09/26
*/
int tlrmvn(int N, const std::vector<Eigen::MatrixXd> &B,
        const std::vector<TLRNode> &UV, const Eigen::VectorXd &lastRow, 
	const Eigen::VectorXd &a1,
        const Eigen::VectorXd &b1, double &v, double &e, int ns, int &scaler_in,
        double *workDbl, int lworkDbl, int *workInt, int lworkInt)
{
        int nb = B.size();
        int m = B[0].rows();
        int n = nb * m;
        int N_tilde = N << 1;
        // dbl work
        double *pN = workDbl;
	double *p0 = pN + N_tilde;
        double *pr = p0 + N_tilde;
        double *y = pr + N_tilde;
        double *values = y + m*N_tilde;
	double *values_numer = values + ns;
        double *a = values_numer + ns;
        double *b = a + n*N_tilde;
	double *aLast = b + n * N_tilde;
	double *bLast = aLast + N_tilde;
        double *qN = bLast + N_tilde;
        double *x = qN + n*N;
        double *xr = x + m*N_tilde;
        double *subworkDbl = xr + m;
        int lsubworkDbl = 6 * N_tilde;
        int memDbl = N_tilde + N_tilde + N_tilde + m*N_tilde + ns + ns + n*N_tilde + 
		n * N_tilde + N_tilde + N_tilde + 
                n * N + m*N_tilde + m + lsubworkDbl;
        if(lworkDbl < memDbl)
                Rcpp::stop("lworkDbl is insufficient\n");
        // int work
        int *prime = workInt;
        int *scaler_N = prime + n;
        int *scaler_ns = scaler_N + N_tilde;
        int *subworkInt = scaler_ns + ns;
        int lsubworkInt = N_tilde;
        int memInt = n + N_tilde + ns + lsubworkInt;
        if(lworkInt < memInt)
                Rcpp::stop("lworkInt is insufficient\n");
        // fixed part of x
        primes(5*n*log((double)n+1)/4, n, prime);
        transform(prime, prime+n, qN, [](int x){return sqrt((double) x);});
        for(int i = 1; i < N; i++)
                transform(qN, qN+n, qN+(i-1)*n, qN+i*n,
                        [](double x1,double x2){return x1 + x2;});
        // some para val
        int sIncr = 1;
        double alpha;
        double beta;
	Map<MatrixXd> yMap(y, m, N_tilde);
	Map<MatrixXd> aMap(a, n, N_tilde);
	Map<MatrixXd> bMap(b, n, N_tilde);
	Map<RowVectorXd> aLastMap(aLast, N_tilde);
	Map<RowVectorXd> bLastMap(bLast, N_tilde);
        // ns batches
        for (int i = 0; i < ns; i++) {
                fill(pN, pN + N_tilde, 1.0);
                fill(scaler_N , scaler_N+N_tilde, 0);
                // replicate the int limits
		for(int j = 0; j < N_tilde; j++)
		{
			copy(a1.data(), a1.data()+n, a+j*n);
			copy(b1.data(), b1.data()+n, b+j*n);
		}
		aLastMap.setConstant(a1(n));
		bLastMap.setConstant(b1(n));
                int offset_UV = 0;
                for (int r = 0; r < nb; r++) {
                        int r1 = r*m;
                        if (r > 0) {
				#ifdef _OPENMP
		                #pragma omp parallel for
		                #endif
                                for(int j = 0 ; j < nb - r ; j++)
                                {
                                        int i1 = r1 + j*m;
                                        int crtColNumUV = UV[offset_UV + j].crtColNum;
					#ifndef _OPENMP
                                        alpha = 1.0;
                                        beta = 0.0;
                                        F77_CALL(dgemm)("T", "N", &crtColNumUV,
                                                &N_tilde, &m, &alpha, UV[offset_UV +
                                                j].V.data(), &m, y, &m, &beta, a, &n);
                                                // use a[1:m, 1:N_tilde] to store VT *
                                                // y
                                        F77_CALL(dgemm)("N", "N", &m, &N_tilde,
                                                &crtColNumUV, &alpha, UV[offset_UV +
                                                j].U.data(), &m, a, &n, &beta, b, &n);
                                                // use b[1:m, 1:N_tilde] to store U *
                                                // a[1:crtColNumUV, 1:N_tilde]
                                        for(int k = 0; k < N_tilde; k++)
                                        {
                                                transform(a+k*n+i1, a+k*n+i1+m, b+k*n,
                                                        a+k*n+i1, [](double x, double
                                                        &y){return x - y;});
                                                transform(b+k*n+i1, b+k*n+i1+m, b+k*n,
                                                        b+k*n+i1, [](double x, double
                                                        &y){return x - y;});
                                        }
					#else
					MatrixXd delta = UV[ offset_UV + j ].U.block(
                                                0,0,m,crtColNumUV) * (UV[
                                                offset_UV + j ].V.block(0,0,m,
                                                crtColNumUV).transpose() * yMap);
                                        aMap.block(i1, 0, m, N_tilde) -= delta;
                                        bMap.block(i1, 0, m, N_tilde) -= delta;
					#endif
                                }
				// update aLast and bLast
				alpha = -1.0;
				beta = 1.0;
				int incr = 1;
				// aLast = aLast - y^T lastRow.segment(r1 - m, m)
				F77_CALL(dgemv)("T", &m, &N_tilde, &alpha, y, 
						&m, lastRow.data() + r1 - m, &incr,
						&beta, aLast, &incr);
				F77_CALL(dgemv)("T", &m, &N_tilde, &alpha, y, 
						&m, lastRow.data() + r1 - m, &incr,
						&beta, bLast, &incr);
				// update offset_UV
                                offset_UV += nb - r;
                        } // if (r > 0)
			GetRNGstate();
                        for_each(xr, xr+m, [] (double &x){x = unif_rand();});
			PutRNGstate();
                        for(int j = 0; j < N; j++)
                                transform(qN+j*n+r1, qN+j*n+r1+m, xr, x+
                                        j*m, [](double &xf, double &xr){return
                                        xf + xr;});
                        for_each(x, x+m*N, [](double &x){x = abs(2.0 * (x - int(x)) -
                                1.0);});
                        transform(x, x+m*N, x+m*N, [](double &x){return 1 - x;});
                        // kernel
                        Map<MatrixXd, 0, OuterStride<>> aBlk(a+r1, m, N_tilde,
                                OuterStride<>(n));
                        Map<MatrixXd, 0, OuterStride<>> bBlk(b+r1, m, N_tilde,
                                OuterStride<>(n));
                        Map<MatrixXd> xMap(x, m, N_tilde);
                        mvndns(m, N_tilde, B[r], xMap, aBlk, bBlk, pr, y, m, scaler_N,
                                subworkDbl, lsubworkDbl, subworkInt, lsubworkInt);
                        // scale
                        transform(pr, pr+N_tilde, pN, pN, [](double &pNew, double
                                pAccu){return pNew*pAccu;});
                        std::transform(pN, pN+N_tilde, subworkInt, [](double &val)
                                {return ilogb(val);});
                        std::for_each(subworkInt, subworkInt+N_tilde, [](int &e)
                                {e = e < MIN_PROB_EXPO ? 0 : e;});
                        std::transform(subworkInt, subworkInt+N_tilde, pN, pN, [](int
                                scaler, double base){return scalbn(base, -scaler);});
                        std::transform(subworkInt, subworkInt+N_tilde, scaler_N,
                                scaler_N, [](int scaler1, int scaler2){
                                return scaler1 + scaler2;});
                } // for(r < nb)
		// update aLast and bLast
		alpha = -1.0;
		beta = 1.0;
		int incr = 1;
		F77_CALL(dgemv)("T", &m, &N_tilde, &alpha, y, 
				&m, lastRow.data() + n - m, &incr,
				&beta, aLast, &incr);
		F77_CALL(dgemv)("T", &m, &N_tilde, &alpha, y, 
				&m, lastRow.data() + n - m, &incr,
				&beta, bLast, &incr);
		// compute the last dimension
		double sdLast = lastRow(n);
		for_each(aLast, aLast + N_tilde, [sdLast](double &x){x = x / sdLast;});
		for_each(bLast, bLast + N_tilde, [sdLast](double &x){x = x / sdLast;});
		lc_vdCdfNorm(N_tilde, aLast, subworkDbl);
		lc_vdCdfNorm(N_tilde, bLast, subworkDbl + N_tilde);
		transform(subworkDbl, subworkDbl + N_tilde, subworkDbl + N_tilde,
			subworkDbl + 2 * N_tilde, [](double &c, double &d){return d - c;});
                // scale
                int scaler_max = *(std::max_element(scaler_N, scaler_N+N_tilde));
                std::transform(scaler_N, scaler_N+N_tilde, pN, pN,
                        [scaler_max](int scaler, double base){return scalbn(base,
                        scaler - scaler_max);});
		// compute the numerator
		transform(pN, pN + N_tilde, subworkDbl + 2 * N_tilde, subworkDbl + 
			3 * N_tilde, [](double &x, double &y){
			return x * y;});
                // store
		values_numer[i] = accumulate(subworkDbl + 3 * N_tilde, subworkDbl + 
			4 * N_tilde, 0.0) / (double) N_tilde;
                values[i] = accumulate(pN, pN+N_tilde, 0.0) / (double) N_tilde; 
                scaler_ns[i] = scaler_max;
        } // i = 0 : ns
        int scaler_max = *(std::max_element(scaler_ns, scaler_ns+ns));
        std::transform(scaler_ns, scaler_ns+ns, values, values,
                [scaler_max](int scaler, double base){return scalbn(base,
                scaler - scaler_max);});
        std::transform(scaler_ns, scaler_ns+ns, values_numer, values_numer,
                [scaler_max](int scaler, double base){return scalbn(base,
                scaler - scaler_max);});
	v = accumulate(values_numer, values_numer + ns, 0.0) /
		accumulate(values, values + ns, 0.0);
	e = 0.0;

        return 0;
}


