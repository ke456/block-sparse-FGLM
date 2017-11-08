#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <linbox/integer.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/matrix/permutation-matrix.h>
#include "linbox/matrix/polynomial-matrix.h"
#include "linbox/algorithms/polynomial-matrix/polynomial-matrix-domain.h"
#include "fflas-ffpack/fflas-ffpack.h"

using namespace LinBox;
using namespace std;

typedef Givaro::Modular<double> GF;
typedef LinBox::PolynomialMatrix<LinBox::PMType::matfirst,LinBox::PMStorage::plain,GF> PMatrix;
typedef LinBox::PolynomialMatrix<LinBox::PMType::polfirst,LinBox::PMStorage::plain,GF> MatrixP;

int main(int argc, char *argv[])
{
	size_t p = 23068673;  // size of the base field
	size_t m = 4;   // row dimension for the blocks
	size_t d = 512; // vector space dimension / dimension of multiplication matrices
	size_t nb = 10; // number of products for timing

	static Argument args[] = {
		{ 'p', "-p p", "Set cardinality of the base field.", TYPE_INT, &p },
		{ 'm', "-m m", "Set the matrix dimension to m.", TYPE_INT,       &m },
		{ 'd', "-d d", "Set degree of matrices to d.", TYPE_INT,  &d },
		{ 'n', "-n nb", "Set the number of iterations of tests to nb.", TYPE_INT,  &nb },
		END_OF_ARGUMENTS
	};	

	parseArguments (argc, argv, args);

	GF field(p);
	long seed = time(NULL);
	typename GF::RandIter rd(field,0,seed);
	PolynomialMatrixMulDomain<GF> PMMD( field );

	//cout << "~~~~~~~~~~~STARTING TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;
	//cout << "base field / number of iterations: " << p << " , " << nb << endl;
	//cout << "dimensions / degrees of matrices : " << m << " x " << m << " x " << m << " , " << d << " x " << d << endl;
	{
		Timer time_init_in, time_init_out, time_product;
		for (size_t i = 0; i < nb; ++i) {
			Timer tm;
			tm.start();
			PMatrix mat1( field, m, m, d+1 );
			PMatrix mat2( field, m, m, d+1 );
			for ( size_t deg=0; deg<mat1.size(); ++deg )
			for ( size_t i=0; i<mat1.rowdim(); ++i )
			for ( size_t j=0; j<mat1.coldim(); ++j )
			{
				rd.random( mat1.ref( i, j, deg ) );
				rd.random( mat2.ref( i, j, deg ) );
			}
			tm.stop();
			time_init_in += tm;
			tm.clear(); tm.start();
			PMatrix prod( field, m, m, mat1.size()+mat2.size()-1 );
			tm.stop(); time_init_out += tm;
			tm.clear(); tm.start();
			PMMD.mul( prod, mat1, mat2 );
			tm.stop(); time_product += tm;
		}
		//cout << "#timing# Initialize matrices: --> " << time_init_in.usertime()/nb << endl;
		//cout << "#timing# Initialize product:  --> " << time_init_out.usertime()/nb << endl;
		//cout << "#timing# Perform product:     --> " << time_product.usertime()/nb << endl;
		cout << d << ", " << time_product.usertime()/nb  << endl;
	}
	//cout << "~~~~~~~~~~~END TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;

	//cout << "~~~~~~~~~~~STARTING TIMINGS MIDPRODUCT~~~~~~~~~~~~~" << endl;
	//cout << "dimensions of matrices : " << 2*m << " x " << 2*m << " x " << m << endl;
	//cout << "degree of matrices : " << d << " x " << 2*d << endl;
	//{
	//	Timer tm;
	//	tm.clear(); tm.start();
	//	size_t s1 = d+1;
	//	size_t s2 = d+1;
	//	PMatrix mat1( field, 2*m, 2*m, s1 );
	//	PMatrix mat2( field, 2*m, m, s1+s2-1 );
	//	for ( size_t deg=0; deg<mat1.size(); ++deg )
	//	for ( size_t i=0; i<mat1.rowdim(); ++i )
	//	for ( size_t j=0; j<mat1.coldim(); ++j )
	//	{
	//		rd.random( mat1.ref( i, j, deg ) );
	//	}
	//	for ( size_t deg=0; deg<mat2.size(); ++deg )
	//	for ( size_t i=0; i<mat2.rowdim(); ++i )
	//	for ( size_t j=0; j<mat2.coldim(); ++j )
	//	{
	//		rd.random( mat2.ref( i, j, deg ) );
	//	}
	//	cout << "#timing# Initialize matrices: --> " << tm.usertime() << endl;
	//	tm.clear(); tm.start();
	//	PMatrix midprod( field, 2*m, m, s2 );
	//	//PolMatDom::PMatrix res2( series.field(), m, n, order2 ); // second residual: midproduct 
	//	tm.stop();
	//	cout << "#timing# Initialize midproduct:  --> " << tm.usertime() << endl;
	//	tm.clear(); tm.start();
	//	PMMD.midproductgen( midprod, mat1, mat2, true, s1, s1+s2 );
	//	//this->_PMMD.midproductgen( res2, approx1, series, true, order1+1, order1+order2 ); // res2 = (approx1*series / X^order1) mod X^order2
	//	tm.stop();
	//	cout << "#timing# Perform midproduct:     --> " << tm.usertime() << endl;
	//	tm.clear(); tm.start();
	//	PMatrix prod( field, 2*m, m, mat1.size()+mat2.size()-1 );
	//	tm.stop();
	//	cout << "#timing# Initialize product:  --> " << tm.usertime() << endl;
	//	tm.clear(); tm.start();
	//	PMMD.mul( prod, mat1, mat2 );
	//	tm.stop();
	//	cout << "#timing# Perform product:     --> " << tm.usertime() << endl;
	//}
	//cout << "~~~~~~~~~~~END TIMINGS MIDPRODUCT~~~~~~~~~~~~~" << endl;



	return 0;
}
