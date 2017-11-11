#include <algorithm>
#include <string>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <linbox/integer.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/matrix/polynomial-matrix.h>
#include <linbox/algorithms/polynomial-matrix/polynomial-matrix-domain.h>
#include "fflas-ffpack/fflas-ffpack.h"

using namespace LinBox;
using namespace std;

typedef Givaro::Modular<double> GF;
typedef LinBox::PolynomialMatrix<LinBox::PMType::matfirst,LinBox::PMStorage::plain,GF> PMatrix;
//typedef LinBox::PolynomialMatrix<LinBox::PMType::polfirst,LinBox::PMStorage::plain,GF> MatrixP;
typedef BlasSubmatrix<BlasMatrix<GF>> View;

void naive_mult( PMatrix & prod, const PMatrix & mat1, const PMatrix & mat2 )
{
	BlasMatrixDomain<GF> BMD( mat1.field() );
	const size_t l = mat1.rowdim();
	const size_t m = mat1.coldim();
	const size_t n = mat2.coldim();
	const size_t d1 = mat1.size()-1;
	const size_t d2 = mat2.size()-1;
	const size_t d = d1+d2;

	BlasMatrix<GF> linmat1( mat1.field(), l*(d1+1), m );
	BlasMatrix<GF> linmat2( mat1.field(), m, n*(d2+1) );

	for ( size_t k = 0; k<=d1; ++k )
	for ( size_t j = 0; j<m; ++j )
	for ( size_t i = 0; i<l; ++i )
		linmat1.setEntry( i+l*k, j, mat1.get( i, j, k ) );

	for ( size_t k = 0; k<=d2; ++k )
	for ( size_t j = 0; j<n; ++j )
	for ( size_t i = 0; i<m; ++i )
		linmat2.setEntry( i, j+n*k, mat2.get( i, j, k ) );

	BlasMatrix<GF> linprod( mat1.field(), l*(d1+1), n*(d2+1) );

	BMD.mul( linprod, linmat1, linmat2 );

	// The following apparently is the bottleneck, is this expected?
	for ( int k=0; k<=(int)d; ++k )
	for (int kk = max(k-(int)d2,0); kk<=min((int)d1,k); ++kk)
	for (size_t j = 0; j < n; ++j)
	for (size_t i = 0; i < l; ++i)
		prod.field().addin( prod.ref( i, j, k ), linprod.getEntry(kk*l+i, (k-kk)*n+j) );
}

void naive_mult2( PMatrix & prod, const PMatrix & mat1, const PMatrix & mat2 )
{
	// assumes prod is correctly initialized (dimension, size)
	// assumes all entries of prod are zero
	BlasMatrixDomain<GF> BMD( mat1.field() );
	const size_t d1 = mat1.size()-1;
	const size_t d2 = mat2.size()-1;
	const size_t d = d1+d2;

	for ( int k=0; k<=(int)d; ++k ) {
		for (int kk = max(k-(int)d2,0); kk<=min((int)d1,k); ++kk) {
			BMD.axpyin( prod[k], mat1[kk], mat2[k-kk] );
		}
	}
}

int main(int argc, char *argv[])
{
	size_t m = 4;   // row dimension for the blocks
	size_t d = 512; // vector space dimension / dimension of multiplication matrices
	size_t nb = 10; // number of products for timing

	size_t p = 23068673;  // size of the base field

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
	BlasMatrixDomain<GF> BMD( field );
	PolynomialMatrixMulDomain<GF> PMMD( field );


	//cout << "~~~~~~~~~~~STARTING TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;
	//cout << "base field / number of iterations: " << p << " , " << nb << endl;
	//cout << "dimensions / degrees of matrices : " << m << " x " << m << " x " << m << " , " << d << " x " << d << endl;
	{ // square polynomial matrix product
		for (size_t i = 0; i < nb; ++i) {
			PMatrix mat1( field, m, m, d+1 );
			PMatrix mat2( field, m, m, d+1 );
			for ( size_t deg=0; deg<mat1.size(); ++deg )
			for ( size_t i=0; i<mat1.rowdim(); ++i )
			for ( size_t j=0; j<mat1.coldim(); ++j )
			{
				rd.random( mat1.ref( i, j, deg ) );
				rd.random( mat2.ref( i, j, deg ) );
			}
			PMatrix prod( field, m, m, mat1.size()+mat2.size()-1 );
			auto start = chrono::system_clock::now();
			PMMD.mul( prod, mat1, mat2 );
			auto end = chrono::system_clock::now();
			cout << m << "," << d << "," << chrono::duration_cast<chrono::microseconds>(end-start).count() << endl;
		}
	}
	//cout << "~~~~~~~~~~~END TIMINGS SQUARE MULT~~~~~~~~~~~~~" << endl;

	//cout << "~~~~~~~~~~~STARTING TIMINGS SQUARE NAIVEMULT~~~~~~~~~~~~~" << endl;
	//cout << "base field / number of iterations: " << p << " , " << nb << endl;
	//cout << "dimensions / degrees of matrices : " << m << " x " << m << " x " << m << " , " << d << " x " << d << endl;
	{ // square polynomial matrix product
		for (size_t i = 0; i < nb; ++i) {
			PMatrix mat1( field, m, m, d+1 );
			PMatrix mat2( field, m, m, d+1 );
			for ( size_t deg=0; deg<mat1.size(); ++deg )
			for ( size_t i=0; i<mat1.rowdim(); ++i )
			for ( size_t j=0; j<mat1.coldim(); ++j )
			{
				rd.random( mat1.ref( i, j, deg ) );
				rd.random( mat2.ref( i, j, deg ) );
			}
			//PMatrix prod1( field, m, m, mat1.size()+mat2.size()-1 );
			//auto start = chrono::system_clock::now();
			//naive_mult( prod1, mat1, mat2 );
			//auto end = chrono::system_clock::now();
			//cout << m << "," << d << "," << chrono::duration_cast<chrono::microseconds>(end-start).count() << endl;

			PMatrix prod2( field, m, m, mat1.size()+mat2.size()-1 );
			auto start = chrono::system_clock::now();
			naive_mult2( prod2, mat1, mat2 );
			auto end = chrono::system_clock::now();
			cout << m << "," << d << "," << chrono::duration_cast<chrono::microseconds>(end-start).count() << endl;
		}
	}
	return 0;
}
