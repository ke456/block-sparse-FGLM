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

	static Argument args[] = {
		{ 'p', "-p p", "Set cardinality of the base field.", TYPE_INT, &p },
		{ 'm', "-m m", "Set the matrix dimension to m.", TYPE_INT,       &m },
		{ 'd', "-d d", "Set degree of matrices to d.", TYPE_INT,  &d },
		END_OF_ARGUMENTS
	};	

	parseArguments (argc, argv, args);

	GF field(p);
	long seed = time(NULL);
	typename GF::RandIter rd(field,0,seed);
	PolynomialMatrixMulDomain<GF> PMMD( field );

	cout << "~~~~~~~~~~~STARTING TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;
	cout << "dimensions of matrices : " << m << " x " << m << " x " << m << endl;
	cout << "degree of matrices : " << d << " x " << d << endl;

	Timer tm;
	tm.clear(); tm.start();
	MatrixP mat1( field, m, m, d+1 );
	MatrixP mat2( field, m, m, d+1 );
	for ( size_t deg=0; deg<mat1.size(); ++deg )
	for ( size_t i=0; i<mat1.rowdim(); ++i )
	for ( size_t j=0; j<mat1.coldim(); ++j )
	{
		rd.random( mat1.ref( i, j, deg ) );
		rd.random( mat2.ref( i, j, deg ) );
	}
	tm.stop();
	cout << "#timing# Initialize matrices: --> " << tm.usertime() << endl;
	tm.clear(); tm.start();
	MatrixP prod( field, m, m, 2*d+1 );
	tm.stop();
	cout << "#timing# Initialize product:  --> " << tm.usertime() << endl;
	tm.clear(); tm.start();
	PMMD.mul( prod, mat1, mat2 );
	tm.stop();
	cout << "#timing# Perform product:     --> " << tm.usertime() << endl;
	cout << "~~~~~~~~~~~END TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;

	//for ( size_t deg=0; deg<mat1.size(); ++ded )
	//for ( size_t i=0; i<mat1.rowdim(); ++i )
	//for ( size_t j=0; j<mat1.coldim(); ++j )
	//	rd.random( mat1.ref( i, j, deg ) );

	return 0;
}
