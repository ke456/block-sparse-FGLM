/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */

#define TIMINGS_ON // to activate timings; note that these may be irrelevant if VERBOSE / EXTRA_VERBOSE are also activated
#define EXTRA_TIMINGS_ON // to activate timings; note that these may be irrelevant if VERBOSE / EXTRA_VERBOSE are also activated
//#define EXTRA_VERBOSE_ON // extra detailed printed objects, like multiplication matrix and polynomial matrices... unreadable except for very small dimensions
//#define VERBOSE_ON // some objects printed for testing purposes, but not the biggest ones (large constant matrices, polynomial matrices..)
//#define NAIVE_ON
#define WARNINGS_ON // comment out if having warnings for heuristic parts is irrelevant --> should probably be 'on'
//#define SPARSITY_COUNT // shows the sparsity of the matrices
// #define TEST_FGLM // testing / timing approximant basis algos
//#define TEST_APPROX // testing / timing approximant basis algos
//#define TEST_KERNEL // testing / timing kernel basis algo
#define TEST_INVARIANT_FACTOR // testing / timing heuristic invariant factor
// #define OUTPUT_FUNC // outputs the computed functions

#include "block-sparse-fglm.h"
#include "all.h"

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <NTL/lzz_pX.h>

using namespace LinBox;
using namespace std;
using namespace NTL;

//-----------------------------------------------//
//-----------------------------------------------//
//        Polynomial matrix stuff                //
//-----------------------------------------------//
//-----------------------------------------------//

//shifts every entry by d
void shift(PolMatDom::PMatrix &result, const PolMatDom::PMatrix &mat, int row, int col, int deg){
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++){
			for (int d = 0; d < deg && (d+deg) < mat(i,j).size(); d++){
				auto element = mat.get(i,j,d+deg);
				result.ref(i,j,d) = element;
			}
		}
}

void mat_to_poly (PolMatDom::Polynomial &p, PolMatDom::MatrixP &mat, int size){
	int actual_size = 0;
	GF::Element zero{0};
	for (int i  = size; i >= 0; i--){
		if (mat.ref(0,0,i) != zero) {
			actual_size = i;
			break;
		}
	}
	p.resize(actual_size+1);
	for (int i = 0; i <= actual_size; i++)
		p[i] = mat.ref(0,0,i);
}

// resize mat to x^(size-1)
void mat_resize (GF &field, PolMatDom::MatrixP &mat, int size){
	PolMatDom::MatrixP temp(field,1,1,size);
	for (int i = 0; i < size; i++){
		temp.ref(0,0,i) = mat.ref(0,0,i);
	}
	mat = temp;
}

template<typename PolMat>
void PolMatDom::print_degree_matrix( const PolMat &pmat ) const {
	const size_t d = pmat.degree();
	for ( size_t i=0; i<pmat.rowdim(); ++i ) {
		for ( size_t j=0; j<pmat.coldim(); ++j ) {
			int deg = d;
			while ( deg>=0 and pmat.get(i,j,deg) == 0 )
				--deg;
			cout << deg << "  ";
		}
		cout << endl;
	}
}

template<typename Matrix>
void PolMatDom::MatrixBerlekampMassey( PolMatDom::PMatrix &mat_gen, PolMatDom::PMatrix &mat_num, const vector<Matrix> & mat_seq, const size_t threshold ) {
	// 0. initialize dimensions, shift, matrices
	size_t M = mat_seq[0].rowdim();
	size_t d = mat_seq.size();
	const vector<int> shift( 2*M, 0 );  // dim = M + N = 2M
	PolMatDom::PMatrix series( this->field(), 2*M, M, d );
	PolMatDom::PMatrix app_bas( this->field(), 2*M, 2*M, d );

	// 1. construct series = Matrix.block( [[sum( [seq[d-k-1] * X^k for k in range(d)] )],[-1]] )
	// i.e. stacking reversed sequence and -Identity
	for ( size_t i=0; i<M; ++i )
		series.ref(i+M,i,0) = this->field().mOne;

	for ( size_t k=0; k<d; ++k )
		for ( size_t i=0; i<M; ++i )
			for ( size_t j=0; j<M; ++j )
				series.ref(i,j,k) = mat_seq[d-k-1].getEntry(i,j);

#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Approximant basis: input order, shift, series degrees" << endl;
	cout << d << endl;
	cout << shift << endl;
	this->print_degree_matrix( series );
#ifdef EXTRA_VERBOSE_ON
	cout << "series entries:" << endl;
	cout << series << endl;
#endif
#endif

	// 2. compute approximant basis in ordered weak Popov form
	vector<size_t> mindeg = this->pmbasis( app_bas, series, d, shift, threshold );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Approximant basis: output rdeg and basis degrees" << endl;
	cout << mindeg << endl;
	this->print_degree_matrix( app_bas );
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif

	// 3. copy into mat_gen and mat_num
	mat_gen.setsize( mindeg[0]+1 );
	mat_num.setsize( mindeg[0]+1 );
	for ( size_t i=0; i<M; ++i )
		for ( size_t j=0; j<M; ++j )
			for ( size_t k=0; k<=mindeg[0]; ++k )
			{
				mat_gen.ref(i,j,k) = app_bas.get(i,j,k);
				mat_num.ref(i,j,k) = app_bas.get(i,j+M,k);
			}
}


/*! \brief Computing s-owP kernel basis of a polynomial matrix
 *
 *  Given an m x n polynomial matrix F, and a shift s = (s1,...,sn),
 *  this function computes a left kernel basis of F in s-ordered
 *  weak Popov form.
 *
 *  Algorithm based on PM-Basis at a high order:
 *  cost bound O~( ? )
 *  --> faster algorithms are known in some cases,
 *  cf. Zhou-Labahn-Storjohann ISSAC 2012 and Vu Thi Xuan's Master 1 report
 *
 * \param kerbas: storage for the output kernel basis
 * \param pmat: input polynomial matrix
 * \param shift: input shift
 * \return the shifted minimal degree of the kernel basis (or the shifted pivot degree and index?)
 */
/* TODO: <vneiger> only supports uniform shift */
/* TODO: <vneiger> specific to our case -> in general might return only a part of the kernel basis */
/* TODO: <vneiger> adjust number of rows in output kernel basis after computation */
void PolMatDom::kernel_basis( PMatrix & kerbas, const PMatrix & pmat, const size_t threshold )
{
	size_t m = pmat.rowdim();
	size_t n = pmat.coldim();
	size_t d = pmat.size()-1;
	size_t order = 1 + floor( (m*d) / (double) (m-n));  // specific to uniform shift; large enough to return whole basis?
	PolMatDom::PMatrix appbas(this->field(), m, m, order );
	PolMatDom::PMatrix series(this->field(), m, n, order );
	for ( size_t k=0; k<=d; ++k )
		for ( size_t j=0; j<n; ++j )
			for ( size_t i=0; i<m; ++i )
				series.ref(i,j,k) = pmat.get(i,j,k);
	const vector<int> shift( m, 0 );
	vector<size_t> mindeg = this->pmbasis( appbas, series, order, shift, threshold );
	kerbas.resize( order-d );
	size_t row = 0;
	for ( size_t i=0; i<m; ++i )
	{
		if (mindeg[i]+d < order)
		{
			for ( size_t j=0; j<m; ++j )
				for ( size_t k=0; k<kerbas.size(); ++k )
					kerbas.ref(row,j,k) = appbas.get(i,j,k);
			++row;
		}
	}
}

void PolMatDom::largest_invariant_factor( vector<zz_pX> & left_multiplier, zz_pX & factor, const PMatrix & pmat, const size_t threshold )
{
	// copy all columns except last one
	PolMatDom::PMatrix subcols( this->field(), pmat.rowdim(), pmat.coldim()-1, pmat.size() );
	for (size_t d = 0; d < pmat.size(); ++d)
	for (size_t i = 0; i < pmat.rowdim(); ++i)
	for (size_t j = 0; j < pmat.coldim()-1; ++j)
		subcols.ref(i,j,d) = pmat.get(i,j,d);

	PolMatDom::PMatrix kerbas( this->field(), 1, pmat.rowdim(), 0 );
	this->kernel_basis( kerbas, subcols, threshold );

	factor = 0;
	zz_pX pol(pmat.size());
	for ( size_t i=0; i<pmat.rowdim(); ++i )
	{
		for (size_t d=0; d<kerbas.size(); ++d)
			SetCoeff(left_multiplier[i], d, (long)kerbas.get(0,i,d));
		for (size_t d=0; d<pmat.size(); ++d)
			SetCoeff(pol, d, (long)pmat.get(i,pmat.coldim()-1,d));

		factor += left_multiplier[i] * pol;
	}
}

void PolMatDom::SmithForm( vector<PolMatDom::Polynomial> &smith, PolMatDom::PMatrix &lfac, PolMatDom::PMatrix &rfac, const PolMatDom::PMatrix &pmat, size_t threshold ) {
	// Heuristic computation of the Smith form and multipliers
	// Algorithm:
	//    - compute left Hermite form hmat1 = umat pmat
	//    - compute right Hermite form hmat2 = hmat1 vmat
	//      (which is Transpose(LeftHermite(Transpose(hmat1)))
	//    - then return (smith,lfac,rfac) = (hmat2,umat,vmat)
	// Note: this is not guaranteed to be correct, but seems to true generically
	// Implementation: Hermite form computed via kernel basis, itself computed via approximant basis
	const size_t M = pmat.rowdim();
	const size_t deg = pmat.degree();

	// build Hermite kernel shift:  [0,...,0,0,M deg, 2 M deg, ..., (M-1) M deg]
	vector<int> shift( 2*M, 0 );
	for ( size_t i=M; i<2*M; ++i ) {
		shift[i] = (i-M)*(deg*M+1);
	}

	// order d such that approximant basis contains kernel basis
	const size_t order = 2*M*deg+1;

	// build series matrix: block matrix with first M rows = pmat; last M rows = -identity
	PolMatDom::PMatrix series( this->field(), 2*M, M, order ); // TODO order-->deg

	for ( size_t k=0; k<=deg; ++k )
		for ( size_t i=0; i<M; ++i )
			for ( size_t j=0; j<M; ++j )
				series.ref(i,j,k) = pmat.get(i,j,k);

	for ( size_t i=M; i<2*M; ++i )
		series.ref(i,i-M,0) = this->field().mOne;

#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### First approximant basis: deg(pmat), input shift, input order..:" << endl;
	cout << deg << endl;
	cout << shift << endl;
	cout << order << endl;
	cout << "series degrees:" << endl;
	this->print_degree_matrix(series);
#ifdef EXTRA_VERBOSE_ON
	cout << "series entries:" << endl;
	cout << series << endl;
#endif
#endif

	// compute first approximant basis
	PolMatDom::PMatrix app_bas( this->field(), 2*M, 2*M, order );
	vector<size_t> mindeg = this->pmbasis( app_bas, series, order, shift, threshold );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### First approximant basis: shifted row degrees and matrix degrees:" << endl;
	cout << mindeg << endl;
	this->print_degree_matrix(app_bas);
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif

	// extract the left factor lfac which is the bottom left block,
	// as well as Transpose(LeftHermite(pmat)) which is the transpose of the bottom right block
	app_bas.resize( order ); // make sure size is order (may have been decreased in approximant basis call)
	PolMatDom::PMatrix series2( this->field(), 2*M, M, order );
	for ( size_t i=0; i<M; ++i )
		for ( size_t j=0; j<M; ++j )
			for ( size_t k=0; k<order; ++k ) {
				lfac.ref(i,j,k) = app_bas.get(i+M,j,k);
				series2.ref(j,i,k) = app_bas.get(i+M,j+M,k);
			}
	// bottom part of series2 = -identity
	for ( size_t i=0; i<M; ++i )
		series2.ref(i+M,i,0) = this->field().mOne;

#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### Second approximant basis: deg(pmat), input shift, input order..:" << endl;
	cout << deg << endl;
	cout << shift << endl;
	cout << order << endl;
	cout << "series degrees:" << endl;
	this->print_degree_matrix(series2);
#ifdef EXTRA_VERBOSE_ON
	cout << "series entries:" << endl;
	cout << series2 << endl;
#endif
#endif

	// compute second approximant basis
	PolMatDom::PMatrix app_bas2( this->field(), 2*M, 2*M, order );
	vector<size_t> mindeg2 = this->popov_pmbasis( app_bas2, series2, order, shift, threshold );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### Second approximant basis: shifted minimal degree and matrix degrees:" << endl;
	cout << mindeg2 << endl;
	this->print_degree_matrix(app_bas2);
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas2 << endl;
#endif
#endif

	// extract the right factor rfac, which is the transpose of the bottom left block
	app_bas2.resize( order ); // make sure size is order (may have been decreased in approximant basis call)
	for ( size_t i=0; i<M; ++i )
		for ( size_t j=0; j<M; ++j )
			for ( size_t k=0; k<order; ++k ) {
				rfac.ref(j,i,k) = app_bas2.get(i+M,j,k);
			}

#ifdef WARNINGS_ON
	// check if this second Hermite form is diagonal
	bool test=true;
	for ( size_t i=1; i<M; ++i )
		for ( size_t j=0; j<i; ++j )
			for ( size_t k=0; k<app_bas2.size(); ++k ) {
				if ( app_bas2.get(i+M,j+M,k) != 0 )
					test=false;
			}
	if ( not test ) {
		cout << "~~~WARNING(Smith)~~~ double Hermite did not yield a diagonal matrix" << endl;
		cout << "   ------->>> the matrix found is not the Smith form. Ask Vincent to add a third Hermite form." << endl;
	}
#endif

	// extract the Smith form
	for ( size_t i=0; i<M; ++i ) {
		smith[i] = app_bas2(i+M,i+M);
		while (!smith[i].empty() && smith[i][smith[i].size()-1] == 0) {
			smith[i].pop_back();
		}
#ifdef WARNINGS_ON
		if ( smith[i].empty() ) {
			cout << "~~~WARNING(Smith)~~~ one of the Smith factors is zero" << endl;
			cout << "   ------->>> something went wrong: check for other warnings above." << endl;
		}
#endif
	}

#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### Degrees of Smith factors:" << endl;
	for ( size_t i=0; i<M; ++i )
		cout << smith[i].size()-1 << "  " ;
	cout << endl;
	cout << "###OUTPUT(Smith)### Degrees in left Smith factor:" << endl;
	this->print_degree_matrix( lfac );
	cout << "###OUTPUT(Smith)### Degrees in right Smith factor:" << endl;
	this->print_degree_matrix( rfac );
#ifdef EXTRA_VERBOSE_ON
	cout << "###OUTPUT(Smith)### Smith factors entries:" << endl;
	cout << smith << endl;
	cout << "###OUTPUT(Smith)### Left Smith factor entries:" << endl;
	cout << lfac << endl;
	cout << "###OUTPUT(Smith)### Right Smith factor entries:" << endl;
	cout << rfac << endl;
#endif
#endif
}


vector<int> PolMatDom::old_pmbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const std::vector<int> &shift, const size_t threshold )
{
	/** Algorithm PM-Basis as detailed in Section 2.2 of
	 *  [Giorgi, Jeannerod, Villard. On the Complexity 
	 *  of Polynomial Matrix Computations. ISSAC 2003]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of degree < order, series to approximate
	 *   - order: positive integer, order of approximation
	 *   - shift: degree shift on the cols of approx
	 *   - threshold: depth for leaves of recursion (when the current order reaches threshold, apply mbasis)
	 **/
	/** Action:
	 *   - Compute and store in 'approx' a shifted minimal approximation basis for (series,order,shift)
	 **/
	/** Output: shifted row degrees of the computed approx **/
	/** Complexity: O(m^w M(order) log(order) ) **/
	/** TODO study the impact of the threshold **/

	if ( order <= threshold )
	{
		std::vector<int> rdeg = old_mbasis( approx, series, order, shift );
		return rdeg;
	}
	else
	{
		size_t m = series.rowdim();
		size_t n = series.coldim();
		size_t order1,order2;
		order1 = order>>1; // order1 ~ order/2
		order2 = order - order1; // order2 ~ order/2, order1 + order2 = order
		vector<int> rdeg( shift );

		PolMatDom::PMatrix approx1( this->field(), m, m, 0 );
		PolMatDom::PMatrix approx2( this->field(), m, m, 0 );

		{
			PolMatDom::PMatrix res1( this->field(), m, n, order1 ); // first residual: series truncated mod X^order1
			res1.copy( series, 0, order1-1 );
			rdeg = old_pmbasis( approx1, res1, order1, rdeg, threshold ); // first recursive call
		} // end of scope: res1 is deallocated here
		{
			PolMatDom::PMatrix res2( series.field(), m, n, order2 ); // second residual: midproduct 
			this->_PMMD.midproductgen( res2, approx1, series, true, order1+1, order1+order2 ); // res2 = (approx1*series / X^order1) mod X^order2
			rdeg = old_pmbasis( approx2, res2, order2, rdeg, threshold ); // second recursive call
		} // end of scope: res2 is deallocated here
		
		// for PMD.mul we need the size to be the sum (even though we have a better bound on the output degree)
		approx.resize( approx1.size()+approx2.size()-1 );
		this->_PMMD.mul( approx, approx2, approx1 );
		// the shifted row degree of approx is rdeg
		//--> bound on deg(approx): max(rdeg)-min(shift) (FIXME a bit pessimistic..)
		int maxdeg = *max_element( rdeg.begin(), rdeg.end() ) - *std::min_element( shift.begin(), shift.end() );
		approx.resize( 1 + min( (int) order, maxdeg ) );
		return rdeg;
	}

}

vector<size_t> PolMatDom::popov_pmbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const std::vector<int> &shift, const size_t threshold )
{
	/** Algorithm Popov-PM-Basis as detailed in
	 *  [Jeannerod, Neiger, Villard. Fast Computation of approximant bases in
	 *  canonical form. Preprint, 2017]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of degree < order, series to approximate
	 *   - order: positive integer, order of approximation
	 *   - shift: degree shift on the cols of approx
	 *   - threshold: depth for leaves of recursion (when the current order reaches threshold, apply mbasis)
	 **/
	/** Action:
	 *   - Compute and store in 'approx' the shift-Popov approximation basis for (series,order)
	 **/
	/** Output: shifted row degrees of the computed approx **/
	/** Complexity: O(m^w M(order) log(order) ) **/

	if ( order <= threshold )
	{
		std::vector<size_t> mindeg = mbasis( approx, series, order, shift );
		return mindeg;
	}
	else
	{
		size_t m = series.rowdim();

		// 1. compute shift-owP approximant basis
		vector<size_t> mindeg( m );
		mindeg = pmbasis( approx, series, order, shift, threshold );
		// TODO could test whether mindeg is already uniform and then avoid second call
		// (and maybe also other trivial cases)

		// 2. compute -mindeg-owP approximant basis
		vector<int> mindegshift( m );
		for ( size_t i=0; i<m; ++i )
			mindegshift[i] = - (int)mindeg[i];
		pmbasis( approx, series, order, mindegshift, threshold );

		// 3. left-multiply by inverse of -mindeg-row leading matrix
		// Note: cdeg(approx) = mindeg
		PolMatDom::MatrixP::Matrix lmat( this->field(), m, m );
		for ( size_t i=0; i<m; ++i )
			for ( size_t j=0; j<m; ++j )
				lmat.setEntry( i, j, approx.get( i, j, mindeg[j] ) );
#ifdef EXTRA_VERBOSE_ON
		cout << "###OUTPUT(popov_pmbasis)### leading matrix of -mindeg-owP app-basis:" << endl;
		cout << lmat << endl;
#endif
		this->_BMD.invin( lmat );
		for ( size_t k=0; k<approx.size(); ++k ) {
			this->_BMD.mulin_right( lmat, approx[k] );
		}
		return mindeg;
	}
}


vector<size_t> PolMatDom::pmbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const std::vector<int> &shift, const size_t threshold )
{
	/** Algorithm PM-Basis as detailed in Section 2.2 of
	 *  [Giorgi, Jeannerod, Villard. On the Complexity 
	 *  of Polynomial Matrix Computations. ISSAC 2003]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of degree < order, series to approximate
	 *   - order: positive integer, order of approximation
	 *   - shift: degree shift on the cols of approx
	 *   - threshold: depth for leaves of recursion (when the current order reaches threshold, apply mbasis)
	 **/
	/** Action:
	 *   - Compute and store in 'approx' a shifted ordered weak Popov
	 *   approximation basis for (series,order,shift)
	 **/
	/* Return:
	 * - the shifted minimal degree for (series,order)
	 */
	/** Output: shifted row degrees of the computed approx **/
	/** Complexity: O(m^w M(order) log(order) ) **/
	/** TODO study the impact of the threshold **/

	if ( order <= threshold )
	{
		vector<size_t> mindeg = mbasis( approx, series, order, shift );
		return mindeg;
	}
	else
	{
		size_t m = series.rowdim();
		size_t n = series.coldim();
		size_t order1,order2;
		order1 = order>>1; // order1 ~ order/2
		order2 = order - order1; // order2 ~ order/2, order1 + order2 = order
		vector<size_t> mindeg( m );

		PolMatDom::PMatrix approx1( this->field(), m, m, 0 );
		PolMatDom::PMatrix approx2( this->field(), m, m, 0 );

		{
			PolMatDom::PMatrix res1( this->field(), m, n, order1 ); // first residual: series truncated mod X^order1
			res1.copy( series, 0, order1-1 );
			mindeg = pmbasis( approx1, res1, order1, shift, threshold ); // first recursive call
		} // end of scope: res1 is deallocated here
		{
			vector<int> rdeg( shift ); // shifted row degrees = mindeg + shift
			for ( size_t i=0; i<m; ++i )
				rdeg[i] += mindeg[i];
			PolMatDom::PMatrix res2( series.field(), m, n, order2 ); // second residual: midproduct 
			this->_PMMD.midproductgen( res2, approx1, series, true, order1+1, order1+order2 ); // res2 = (approx1*series / X^order1) mod X^order2
			vector<size_t> mindeg2( m );
			mindeg2 = pmbasis( approx2, res2, order2, rdeg, threshold ); // second recursive call
			for ( size_t i=0; i<m; ++i )
				mindeg[i] += mindeg2[i];
		} // end of scope: res2 is deallocated here
		
		// for PMD.mul we need the size to be the sum (even though we have a better bound on the output degree)
		//approx.resize( approx1.size()+approx2.size()-1 );
		// in fact, deg(approx) = max(mindeg)  (which is indeed the sum of the mindegs for approx1 and approx2)
		approx.resize( 1 + *max_element( mindeg.begin(), mindeg.end() ) );
		this->_PMMD.mul( approx, approx2, approx1 );
		return mindeg;
	}

}



vector<int> PolMatDom::old_mbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const vector<int> &shift )
{
	/** Algorithm M-Basis as detailed in Section 2.1 of
	 *  [Giorgi, Jeannerod, Villard. On the Complexity 
	 *  of Polynomial Matrix Computations. ISSAC 2003]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of degree < order, series to approximate
	 *   - order: positive integer, order of approximation
	 *   - shift: degree shifts on the cols of approx
	 **/
	/** Action:
	 *   - Compute and store in 'approx' a minimal shifted approximation basis for (series,order,shift)
	 **/
	/** Output: shifted row degrees of the computed approx **/
	/** Complexity: O(m^w order^2) **/

	bool resUpdate = true; // true: continuously update the residual. false: compute the residual from approx and series. //FIXME should be argument

	const size_t m = series.rowdim();
	const size_t n = series.coldim();
	typedef BlasSubmatrix<typename PolMatDom::MatrixP::Matrix> View;

	// allocate some space for approx:
	//FIXME right now size = order (pessimistic in many usual cases)
	// maybe an idea: first, twice the expected degree (generic instance),
	//will be increased later if necessary
	// but still this pessimistic version doesn't seem to really impact the timing.. ??
	size_t size_app = order+1;
	approx.resize(0); // to put zeroes everywhere.. FIXME may be a better way to do it but it seems approx.clear() fails
	approx.resize( size_app );

	// set approx to identity, initial degree of approx = 0	
	for ( size_t i=0; i<m; ++i )
		approx.ref(i,i,0) = 1;
	size_t maxdeg = 0; // actual max degree in the matrix

	// initial row shifted degrees = shift - min(shift)
	//this way, we have during all the algo the upper bound max(rdeg) on the degree of approx
	vector<size_t> rdeg( m );
	int min_shift = *min_element( shift.begin(), shift.end() );
	for ( size_t i=0; i<m; ++i )
		rdeg[i] = (size_t) (shift[i] - min_shift);

	// set residual to input series
	PolMatDom::PMatrix res( this->field(), m, n, series.size() );
	res.copy( series );

	for ( size_t ord=0; ord<order; ++ord )
	{
		//At the beginning of iteration 'ord',
		//   - approx is an order basis, shift-reduced,
		//   for series at order 'ord'
		//   - the shift-min(shift) row degrees of approx are rdeg.
		//   - the max degree in approx is <= maxdeg

		// permutation which sorts the shifted row degrees increasingly
		vector<size_t> perm_rdeg( m );
		for ( size_t i=0; i<m; ++i )
			perm_rdeg[i] = i;
		sort(perm_rdeg.begin(), perm_rdeg.end(),
		     [&](const size_t& a, const size_t& b)->bool
		     {
			     return (rdeg[a] < rdeg[b]);
		     } );

		// permute rows of res and approx accordingly, as well as row degrees
		vector<size_t> lperm_rdeg( m ); // LAPACK-style permutation
		FFPACK::MathPerm2LAPACKPerm( lperm_rdeg.data(), perm_rdeg.data(), m );
		BlasPermutation<size_t> pmat_rdeg( lperm_rdeg );
		if ( resUpdate )
		{
			for ( size_t d=ord; d<res.size(); ++d )
				this->_BMD.mulin_right( pmat_rdeg, res[d] );
		}
		for ( size_t d=0; d<=maxdeg; ++d )
			this->_BMD.mulin_right( pmat_rdeg, approx[d] );
		vector<size_t> old_rdeg( rdeg );
		for ( size_t i=0; i<m; ++i )
			rdeg[i] = old_rdeg[perm_rdeg[i]];
		
		// coefficient of degree 'ord' of residual:
		//we aim at cancelling this matrix with a degree 1 polynomial matrix
		typename PolMatDom::MatrixP::Matrix res_const( approx.field(), m, n );

		if ( resUpdate ) // res_const is coeff of res of degree ord
			res_const = res[ord];
		else // res_const is coeff of approx*res of degree ord
		{
			for ( size_t d=0; d<=maxdeg; ++d ) // FIXME using midproduct may (should) be faster?
				this->_BMD.axpyin( res_const, approx[d], res[ord-d] ); // note that d <= maxdeg <= ord
		}
		
		// compute PLUQ decomposition of res_const
		BlasPermutation<size_t> P(m), Q(n);
		size_t rank = FFPACK::PLUQ( res_const.field(), FFLAS::FflasNonUnit, //FIXME TODO investigate see below ftrsm
					    m, n, res_const.getWritePointer(), res_const.getStride(),
					    P.getWritePointer(), Q.getWritePointer() );

		// compute a part of the left kernel basis of res_const:
		// -Lbot Ltop^-1 , stored in Lbot
		// Note: the full kernel basis is [ -Lbot Ltop^-1 | I ] P
		View Ltop( res_const, 0, 0, rank, rank ); // top part of lower triangular matrix in PLUQ
		View Lbot( res_const, rank, 0, m-rank, rank ); // bottom part of lower triangular matrix in PLUQ
		FFLAS::ftrsm( approx.field(), FFLAS::FflasRight, FFLAS::FflasLower,
			      FFLAS::FflasNoTrans, FFLAS::FflasUnit, // FIXME TODO works only if nonunit in PLUQ and unit here; or converse. But not if consistent...?????? investigate
			      approx.rowdim()-rank, rank, approx.field().mOne,
			      Ltop.getPointer(), Ltop.getStride(),
			      Lbot.getWritePointer(), Lbot.getStride() );

		// Prop: this "kernel portion" is now stored in Lbot.
		//Then const_app = [ [ X Id | 0 ] , [ Lbot | Id ] ] P
		//is an order basis in rdeg-Popov form for const_res at order 1
		// --> by transitivity,  const_app*approx will be an order basis
		//for (series,ord+1,shift)

		// update approx basis: 1/ permute all the rows; multiply by constant
		for ( size_t d=0; d<=maxdeg; ++d )
			this->_BMD.mulin_right( P, approx[d] ); // permute rows by P

		for ( size_t d=0; d<=maxdeg; ++d )
		{ // multiply by constant: appbot += Lbot apptop
			View apptop( approx[d], 0, 0, rank, approx.coldim() );
			View appbot( approx[d], rank, 0, approx.rowdim()-rank, approx.coldim() );
			this->_BMD.axpyin( appbot, Lbot, apptop );
		}

		// permute row degrees accordingly
		vector<size_t> lperm_p( P.getStorage() ); // Lapack-style permutation P
		vector<size_t> perm_p( m ); // math-style permutation P
		FFPACK::LAPACKPerm2MathPerm( perm_p.data(), lperm_p.data(), m ); // convert
		vector<size_t> old_rdeg_bis( rdeg );
		for ( size_t i=0; i<rank; ++i ) // update rdeg: rows <rank will be multiplied by X
			rdeg[i] = old_rdeg_bis[ perm_p[i] ] + 1;
		for ( size_t i=rank; i<rdeg.size(); ++i ) // update rdeg: wrdeg of rows >=rank is unchanged
			rdeg[i] = old_rdeg_bis[ perm_p[i] ];

		// compute new max degree
		// Beware: deg(approx) bounded by max(rdeg) BECAUSE we ensured the corresponding shift satisfies min(shift)=0
		// FIXME slightly pessimistic.. do we care? make sure almost not pessimistic in generic case
		maxdeg = min( ord+1, *max_element( rdeg.begin(), rdeg.end() ) );

		// update approx basis: 2/ multiply first rank rows by X...
		for ( size_t d=maxdeg; d>0; --d )
			for ( size_t i=0; i<rank; ++i )
				for ( size_t j=0; j<approx.coldim(); ++j )
					approx.ref(i,j,d) = approx.ref(i,j,d-1);
		// ... and approx[0]: first rank rows are zero
		for ( size_t i=0; i<rank; ++i )
			for ( size_t j=0; j<approx.coldim(); ++j )
				approx.ref(i,j,0) = 0;

		if ( resUpdate )
		{
			// update residual: do same operations as on approx
			// update residual: 1/ permute all the rows; multiply by constant
			// note: to simplify later multiplication by X, we permute rows of res[ord]
			//(but we don't compute the zeroes in the other rows)
			this->_BMD.mulin_right( P, res[ord] ); // permute rows by P
			for ( size_t d=ord+1; d<res.size(); ++d )
				this->_BMD.mulin_right( P, res[d] ); // permute rows by P

			for ( size_t d=ord+1; d<res.size(); ++d )
			{
				// multiply by constant: resbot += Lbot restop
				View restop( res[d], 0, 0, rank, res.coldim() );
				View resbot( res[d], rank, 0, res.rowdim()-rank, res.coldim() );
				this->_BMD.axpyin( resbot, Lbot, restop );
			}

			// update residual: 2/ multiply first rank rows by X...
			for ( size_t d=res.size()-1; d>ord; --d )
				for ( size_t i=0; i<rank; ++i )
					for ( size_t j=0; j<res.coldim(); ++j )
						res.ref(i,j,d) = res.ref(i,j,d-1);
		}
	}

	approx.resize( maxdeg+1 ); // TODO put before update approx basis and remove initial pessimistic allocation
	// now we shift back to the original shift (note approx has no zero row)
	vector<int> rdeg_out( m );
	for ( size_t i=0; i<m; ++i )
		rdeg_out[i] = min_shift + (int) rdeg[i];

	return rdeg_out;
}


vector<size_t> PolMatDom::mbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const vector<int> &shift, bool resUpdate )
{
	/** Algorithm M-Basis-One as detailed in Section 3 of
	 *  [Jeannerod, Neiger, Villard. Fast Computation of approximant bases in
	 *  canonical form. Preprint, 2017]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of size = order, series to approximate
	 *   - order: positive integer, order of approximation
	 *   - shift: m-tuple of degree shifts (acting on columns of approx)
	 **/
	/** Action:
	 *   - Compute and store in 'approx' an shift-ordered weak Popov
	 *   approximation basis for (series,order)
	 *  Note: is order=1, then approx is in shift-Popov form
	 **/
	/** Output: shifted minimal degree of (series,order),
	 * which is equal to the diagonal degrees of approx **/
	/** Complexity: O(m^w order^2) **/

	const size_t m = series.rowdim();
	const size_t n = series.coldim();
	typedef BlasSubmatrix<typename PolMatDom::MatrixP::Matrix> View;

	// initialize approx to the identity matrix
	approx.resize(0); // to put zeroes everywhere.. FIXME may be a better way to do it but it seems approx.clear() fails
	size_t appsz = 1;
	approx.resize(appsz);
	for ( size_t i=0; i<m; ++i )
		approx.ref(i,i,0) = 1;

	// initial shifted row degrees = shift
	vector<int> rdeg( shift );
	// initial shifted minimal degree = (0,...,0)
	vector<size_t> mindeg( m, 0 );

	// set residual to input series
	PolMatDom::PMatrix res( this->field(), m, n, series.size() );
	res.copy( series );

	for ( size_t ord=0; ord<order; ++ord )
	{
		//At the beginning of iteration 'ord',
		//   - approx is an order basis, shift-ordered weak Popov,
		//   for series at order 'ord'
		//   - the shift-min(shift) row degrees of approx are rdeg.
		//   - the max degree in approx is <= appsz

		// Here we follow [Algorithm M-Basis-1] in the above reference

		// coefficient of degree 'ord' of residual, which we aim at cancelling
		typename PolMatDom::MatrixP::Matrix res_const( approx.field(), m, n );
		if ( resUpdate ) // res_const is coeff of res of degree ord
			res_const = res[ord];
		else // res_const is coeff of approx*res of degree ord
		{
			for ( size_t d=0; d<appsz; ++d )
				this->_BMD.axpyin( res_const, approx[d], res[ord-d] ); // note that d <= appsz-1 <= ord
		}

		// permutation for the stable sort of the shifted row degrees
		vector<size_t> perm_rdeg( m );
		iota( perm_rdeg.begin(), perm_rdeg.end(), 0 );
		stable_sort(perm_rdeg.begin(), perm_rdeg.end(),
			    [&](const size_t& a, const size_t& b)->bool
			    {
				    return (rdeg[a] < rdeg[b]);
			    } );

		// permute rows of res_const accordingly
		vector<size_t> lperm_rdeg( m ); // LAPACK-style permutation
		FFPACK::MathPerm2LAPACKPerm( lperm_rdeg.data(), perm_rdeg.data(), m );
		BlasPermutation<size_t> pmat_rdeg( lperm_rdeg );
		this->_BMD.mulin_right( pmat_rdeg, res_const );

		// compute PLUQ decomposition of res_const
		BlasPermutation<size_t> P(m), Q(n);
		size_t rank = FFPACK::PLUQ( res_const.field(), FFLAS::FflasNonUnit, //FIXME TODO investigate see below ftrsm
					    m, n, res_const.getWritePointer(), res_const.getStride(),
					    P.getWritePointer(), Q.getWritePointer() );

		// compute a part of the left kernel basis of res_const:
		// -Lbot Ltop^-1 , stored in Lbot
		// Note: the full kernel basis is [ -Lbot Ltop^-1 | I ] P
		View Ltop( res_const, 0, 0, rank, rank ); // top part of lower triangular matrix in PLUQ
		View Lbot( res_const, rank, 0, m-rank, rank ); // bottom part of lower triangular matrix in PLUQ
		FFLAS::ftrsm( approx.field(), FFLAS::FflasRight, FFLAS::FflasLower,
			      FFLAS::FflasNoTrans, FFLAS::FflasUnit, // FIXME TODO works only if nonunit in PLUQ and unit here; or converse. But not if consistent...?????? investigate
			      m-rank, rank, approx.field().mOne,
			      Ltop.getPointer(), Ltop.getStride(),
			      Lbot.getWritePointer(), Lbot.getStride() );

		// Prop: this "kernel portion" is now stored in Lbot.
		//Then const_app = perm^{-1} P^{-1} [ [ X Id | 0 ] , [ Lbot | Id ] ] P perm
		//is an order basis in rdeg-Popov form for const_res at order 1
		// --> by transitivity,  const_app*approx will be a shift-ordered
		// weak Popov approximant basis for (series,ord+1)

		// A. update approx basis, first steps:
		for ( size_t d=0; d<appsz; ++d )
		{
			// 1. permute rows: approx = P * perm * approx
			this->_BMD.mulin_right( pmat_rdeg, approx[d] );
			this->_BMD.mulin_right( P, approx[d] );
			// 2. multiply by constant: appbot += Lbot apptop
			View apptop( approx[d], 0, 0, rank, m );
			View appbot( approx[d], rank, 0, m-rank, m );
			this->_BMD.axpyin( appbot, Lbot, apptop );
		}

		// permute row degrees accordingly
		vector<size_t> lperm_p( P.getStorage() ); // Lapack-style permutation P
		vector<size_t> perm_p( m ); // math-style permutation P
		FFPACK::LAPACKPerm2MathPerm( perm_p.data(), lperm_p.data(), m ); // convert

		// B. update shifted row degree, shifted minimal degree,
		// and new approximant basis size using property: deg(approx) = max(mindeg)
		for ( size_t i=0; i<rank; ++i )
		{
			++rdeg[perm_rdeg[perm_p[i]]];
			++mindeg[perm_rdeg[perm_p[i]]];
		}
		appsz = 1 + *max_element(mindeg.begin(),mindeg.end());
		approx.resize( appsz );

		// A. update approx basis:
		// 3. multiply first rank rows by X...
		for ( size_t d=appsz-1; d>0; --d )
			for ( size_t i=0; i<rank; ++i )
				for ( size_t j=0; j<m; ++j )
					approx.ref(i,j,d) = approx.ref(i,j,d-1);
		// 4. ... and approx[0]: first rank rows are zero
		for ( size_t i=0; i<rank; ++i )
			for ( size_t j=0; j<m; ++j )
				approx.ref(i,j,0) = 0;
		// 5. permute the rows again: approx = perm^{-1} * P^{-1} * approx
		P.Invert();
		pmat_rdeg.Invert();
		for ( size_t d=0; d<appsz; ++d )
		{
			this->_BMD.mulin_right( P, approx[d] );
			this->_BMD.mulin_right( pmat_rdeg, approx[d] );
		}

		// TODO work on resUpdate = True; now assuming False
		if ( resUpdate )
		{
			//if ( resUpdate )
			//{
			//	for ( size_t d=ord; d<res.size(); ++d )
			//		this->_BMD.mulin_right( pmat_rdeg, res[d] );
			//}
			// update residual: do same operations as on approx
			// update residual: 1/ permute all the rows; multiply by constant
			// note: to simplify later multiplication by X, we permute rows of res[ord]
			//(but we don't compute the zeroes in the other rows)
			this->_BMD.mulin_right( P, res[ord] ); // permute rows by P
			for ( size_t d=ord+1; d<res.size(); ++d )
				this->_BMD.mulin_right( P, res[d] ); // permute rows by P

			for ( size_t d=ord+1; d<res.size(); ++d )
			{
				// multiply by constant: resbot += Lbot restop
				View restop( res[d], 0, 0, rank, n );
				View resbot( res[d], rank, 0, m-rank, n );
				this->_BMD.axpyin( resbot, Lbot, restop );
			}

			// update residual: 2/ multiply first rank rows by X...
			for ( size_t d=res.size()-1; d>ord; --d )
				for ( size_t i=0; i<rank; ++i )
					for ( size_t j=0; j<n; ++j )
						res.ref(i,j,d) = res.ref(i,j,d-1);
		}
	}
	return mindeg;
}

//-----------------------------------------------//
//-----------------------------------------------//
//  Reading input data from file                 //
//-----------------------------------------------//
//-----------------------------------------------//
InputMatrices::InputMatrices(string & filename):
	filename(filename)
{

	string line;
	ifstream file;
	file.open (filename);
	// TODO: what is the file format?
	getline(file, name);
	getline(file, line);
	p = stoi(line);
	getline(file, line);
	n = stoi(line);
	getline(file, line);
	D = stoi(line);

	x = vector< vector<int> >(n, vector<int>(0));
	y = vector< vector<int> >(n, vector<int>(0));
	data = vector< vector<double> >(n, vector<double>(0));

	long max_entries = D*D;
	int index = 0;
	long count = 0;
	while (getline(file, line)){
		istringstream sline(line);
		vector<string> numbers{istream_iterator<string>{sline}, istream_iterator<string>{}};
		int i = stoi(numbers[0]);
		int j = stoi(numbers[1]);
		int a_int = stoi(numbers[2]);

		if (i == D){
			index++;
			sparsity.emplace_back((double)(count)/max_entries);
			count = 0;
		}
		else{
			x[index].emplace_back(i);
			y[index].emplace_back(j);
			data[index].emplace_back(a_int);
			count++;
		}
	}
	file.close();
}

//reads matrices from s
Block_Sparse_FGLM::Block_Sparse_FGLM(size_t M, InputMatrices& mat, size_t threshold):
	field(GF(mat.p)),
	prime(mat.p),
	D(mat.D),
        M(M),
	n(mat.n),
        threshold(threshold),
	mul_mats(n+1, SparseMatrix<GF>(field, D, D)),
//	V(field, D, M),
	sparsity(mat.sparsity),
	name(mat.name),
	filename(mat.filename),
	ofs(name + "_sol.sage")
{
	
	srand(0);
	// srand(time(NULL));
	for (int i = 0; i < n; i++){
		GF::Element a;
		field.init(a,rand());
		rand_comb.emplace_back(a);
	}

	for (int i = 0; i < n; i++)
		for (int j = 0; j < mat.x[i].size(); j++){
			int xx = mat.x[i][j];
			int yy = mat.y[i][j];
			GF::Element a(mat.data[i][j]);
			mul_mats[i].refEntry(xx, yy) = a;

			GF::Element e;
			field.mul(e, rand_comb[i], a);
			field.add(e, e, mul_mats[n].refEntry(xx, yy));
			mul_mats[n].refEntry(xx, yy) = e;
		}

	
#ifdef SPARSITY_COUNT
	cout << "sparsity: ";
	for (auto i: sparsity)
		cout << i << " ";
	cout << endl;
#endif
#ifdef EXTRA_VERBOSE_ON
	for (auto &i : mul_mats)
		i.write(cout << "mul mat", Tag::FileFormat::Maple)<<endl;
#endif

	zz_p::init(prime);
}


template<typename Matrix>
void Block_Sparse_FGLM::create_random_matrix(Matrix &m){
	GF::Element a;
	for (size_t i = 0; i < m.rowdim(); i++)
		for (size_t j = 0; j < m.coldim(); j++){
			field.init(a,rand());
			m.refEntry(i,j) = a; 
		}
}

//-------------------------------------------------------//
// puts U*mul_mats[numvar]^i into v, i=0...2D/M          //
// v_flat is v flattened into a single matrix            //
// numvar = n means we are using a random combination    //
//-------------------------------------------------------//
void Block_Sparse_FGLM::get_matrix_sequence_left(DenseMatrix<GF> &v_flat, int numvar){
	MatrixDomain<GF> MD{field};

	// initialize the left block (random MxD)
	vector<DenseMatrix<GF>> v = vector<DenseMatrix<GF>>(getLength(), DenseMatrix<GF>(field, M, D));

// ??????????????????????? we create U here???????????????????????
	vector<DenseMatrix<GF>> U_rows(M, DenseMatrix<GF>(field, 1, D));
	for (auto &i : U_rows){
		create_random_matrix(i);
#ifdef EXTRA_VERBOSE_ON
		i.write(cout << "###OUTPUT### U_row: ", Tag::FileFormat::Maple) << endl;
#endif
	}
	



	// stores U_i*T1^j at mat_seq[i][j]
	vector<vector<DenseMatrix<GF>>> mat_seq(M);
	for (auto &i : mat_seq)
		i = vector<DenseMatrix<GF>>(getLength(), DenseMatrix<GF>(field, 1, D));


	// initialize the first multiplication matrix
	SparseMatrix<GF> *T1 = &mul_mats[numvar];
#ifdef EXTRA_VERBOSE_ON
	T1->write(cout << "###OUTPUT### Multiplication matrix T:" << endl, Tag::FileFormat::Maple) << endl;
#endif

	// compute sequence in a parallel fashion
#pragma omp parallel for num_threads(M)
	for (int i  = 0; i < M; i++){
		MatrixDomain<GF> MD2{field};
		vector<DenseMatrix<GF>> temp_mat_seq(getLength(), DenseMatrix<GF>(field, 1, D)); 
		temp_mat_seq[0] = U_rows[i];
		for (size_t j  = 1; j < getLength(); j++){
			MD2.mul(temp_mat_seq[j], temp_mat_seq[j-1], *T1);
		}
		mat_seq[i] = temp_mat_seq;
	}
	
	for (size_t i = 0; i < getLength(); i++){
		auto &temp = v[i];
		for (int row = 0; row < M; row++){
			for (int col = 0; col < D; col++){
				GF::Element a;
				mat_seq[row][i].getEntry(a, 0, col);
				temp.refEntry(row, col) = a;
			}
		}
	} 

	// gather all the matrices of l in a single (seq_length*M) x D matrix
	v_flat = DenseMatrix<GF>(field, getLength()*M, D);
	for (size_t i = 0; i < getLength(); i++){
		auto &m = v[i];
		for (int row = 0; row < M; row++){
			int r = i * M + row; // starting point for mat
			for (int col = 0; col < D; col++){
				GF::Element a;
				m.getEntry(a,row, col);
				v_flat.refEntry(r, col) = a;
			}
		}
	}
}

//-------------------------------------------------------//
// Computes sequence v = (UT1^i)V                        //
// assumes that v_flat holds the matrix of all (U T1^i)  //
//-------------------------------------------------------//
void Block_Sparse_FGLM::get_matrix_sequence(vector<DenseMatrix<GF>> &v, 
					    const DenseMatrix<GF> &l_flat, 
					    const DenseMatrix<GF> &V, int c, size_t to){

	// multiplication
	MatrixDomain<GF> MD(field);
	DenseMatrix<GF> result(field, to*M, c);
	MD.mul(result, l_flat, V);

	v = vector<DenseMatrix<GF>>(getLength(), DenseMatrix<GF>(field, M, M));
	v.resize(to, DenseMatrix<GF>(field, M, c));
	for (size_t i = 0; i < to; i++){
		v[i] = DenseMatrix<GF>(field, M, c);
		for (int row = 0; row < M; row++){
			int r = i * M + row;
			for (int col = 0; col < c; col++){
				GF::Element a;
				result.getEntry(a, r, col);
				v[i].refEntry(row, col) = a;
			}
		}
	}
}


//-------------------------------------------------------//
//-------------------------------------------------------//
void Block_Sparse_FGLM::Omega(zz_pX & numerator, const PolMatDom::MatrixP &u_tilde,
			      const PolMatDom::PMatrix &mat_gen,
			      const DenseMatrix<GF> &mat_seq_left_flat,
			      const DenseMatrix<GF> &right_mat){

	PolMatDom PMD(field);
	PolynomialMatrixMulDomain<GF> PMMD(field);

	vector<DenseMatrix<GF>> seq(getGenDeg(), DenseMatrix<GF>(field, M, 1));
	get_matrix_sequence(seq, mat_seq_left_flat, right_mat, 1, getGenDeg());


	PolMatDom::PMatrix polys(PMD.field(), M, 1, getGenDeg());
	// creating the poly matrix from the sequence in reverse order
	int index = 0;
	for (int j = seq.size()-1; j >= 0; j--){
		for (int q = 0; q < M; q++){
			auto element = seq[j].refEntry(q, 0);
			polys.ref(q, 0, index) = element;
		}
		index++;
	}
	PolMatDom::PMatrix N(PMD.field(), M, 1, 1+getLength());
	PolMatDom::PMatrix N_shift(PMD.field(), M, 1, 1+getLength());
	for (long j = 0; j < M; j++)
		for (long i = 0; i < (1+getLength()); i++){
			N.ref(j, 0, i) = 0;
		}
	for (long j = 0; j < M; j++)
		for (long i = 0; i < 1+getLength(); i++){
			N_shift.ref(j, 0, i) = 0;
		}

	PMMD.mul(N, mat_gen, polys);
	shift(N_shift, N, M, 1, getGenDeg());

	PolMatDom::MatrixP n_mat(PMD.field(), 1, 1, D+1);
	for (long i = 0; i < (D+1); i++)
		n_mat.ref(0, 0, i) = 0;
	PMMD.mul(n_mat, u_tilde, N_shift);
	mat_resize(field, n_mat, D);

	numerator = 0;
	for (int i  = 0; i < D; i++)
		SetCoeff(numerator, i, (long)n_mat.get(0, 0, i));
}

//--------------------------------------------------------------------------------//
// builds a random V                                                              //
// applies matrix-BM and smith                                                    //
// return u_tilde, the minimal matrix generator and the sqfree part of minpoly    //
//--------------------------------------------------------------------------------//
void Block_Sparse_FGLM::smith(PolMatDom::MatrixP &u_tilde, PolMatDom::PMatrix &mat_gen, zz_pX &min_poly_sqfree, 
			      zz_pX &min_poly_multiple, const DenseMatrix<GF> &mat_seq_left_flat){

#ifdef TIMINGS_ON
	Timer tm;
	tm.clear(); 
	tm.start();
#endif
	//---------------------------------------
	// 1. compute the whole matrix sequence
	//---------------------------------------
	DenseMatrix<GF> V = DenseMatrix<GF>(field, D, M); //right side of U*T1*V
	create_random_matrix(V);
	vector<DenseMatrix<GF>> mat_seq;
	get_matrix_sequence(mat_seq, mat_seq_left_flat, V, M, getLength());

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### matrix_sequence: " << tm.usertime() << endl; 
	tm.clear(); 
	tm.start();
#endif

	//---------------------------------------
	// 2. matrix Berlekamp-Massey
	//---------------------------------------
	PolMatDom PMD(field);
	mat_gen = PolMatDom::PMatrix (PMD.field(), M, M, getLength());
	PolMatDom::PMatrix mat_num(PMD.field(), M, M, getLength());

 	PMD.MatrixBerlekampMassey<DenseMatrix<GF>>( mat_gen, mat_num, mat_seq, getThreshold() );

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Matrix Berlekamp Massey: " << tm.usertime() << endl; 
#endif
#ifdef VERBOSE_ON
  	cout << "###OUTPUT### Matrix generator degrees:" << endl;
	PMD.print_degree_matrix( mat_gen );
  	cout << "###OUTPUT### Matrix numerator degrees:" << endl;
	PMD.print_degree_matrix( mat_num );
#endif
#ifdef EXTRA_VERBOSE_ON
  	cout << "###OUTPUT### Matrix generator entries:" << endl;
	cout << mat_gen << endl;
  	cout << "###OUTPUT### Matrix numerator entries:" << endl;
	cout << mat_num << endl;
#endif
#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	//---------------------------------------
	// 3. smith form and transformations
	//---------------------------------------
	vector<PolMatDom::Polynomial> smith( M );
	PolMatDom::PMatrix lfac(PMD.field(), M, M, M*this->getLength()+1);
	PolMatDom::PMatrix rfac(PMD.field(), M, M, M*this->getLength()+1);
	PMD.SmithForm( smith, lfac, rfac, mat_gen, this->getThreshold() );
	vector<zz_pX> zz_pX_smith (M);
	for (int i = 0; i < M; i++){
		for (int j = 0; j < smith[i].size(); j++)
			SetCoeff(zz_pX_smith[i], j, (long)smith[i][j]);
		zz_pX_smith[i].normalize();
	}

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Smith form and transformations: " << tm.usertime() << endl; 
	tm.clear();
	tm.start();
#endif

	//---------------------------------------
	// 4. finding u_tilde
	//---------------------------------------
	// Making a matrix with just minpoly as the entry
	PolMatDom::MatrixP P_mat(PMD.field(), 1, 1, D+1);
	for (int i = 0; i < smith[0].size(); i++){
		auto element = smith[0][i];
		P_mat.ref(0, 0, i) = element;
	}

	PolynomialMatrixMulDomain<GF> PMMD(field);
	PolMatDom::MatrixP rfac_row(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::MatrixP result(PMD.field(), 1, M, M*getLength()+1);
	
	// extracting the first row of rfac
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M*getLength()+1; j++){
			auto element = rfac.get(0, i, j);
			rfac_row.ref(0, i, j) = element;
		}
	PMMD.mul(result, P_mat, rfac_row);

	vector<zz_pX> rfac_polys(M);
	vector<zz_pX> div(M);
	for (int i = 0; i < M; i++){
		for (int j = 0; j < M*getLength()+1; j++)
			SetCoeff(rfac_polys[i], j, (long)result.get(0, i, j));
		rfac_polys[i].normalize();
		div[i] = rfac_polys[i] / zz_pX_smith[i];
	}
	PolMatDom::MatrixP w(PMD.field(), 1, M, M*getLength()+1);
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M*getLength()+1; j++)
			w.ref(0, i, j) = coeff(div[i], j)._zz_p__rep;
  
	u_tilde = PolMatDom::MatrixP(PMD.field(), 1, M, D+3);
	PMMD.mul(u_tilde, w, lfac);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Computing u_tilde: " << tm.usertime() << endl;;
	tm.clear();
	tm.start();
#endif

	//---------------------------------------
	// 5. finding the min poly
	//---------------------------------------
	zz_pX min_poly;
	for (int i = 0; i < smith[0].size(); i++)
		SetCoeff(min_poly, i, (long)smith[0][i]);
	cout << "minpoly\n" << min_poly << endl;
	zz_pX dM = diff(min_poly);
	min_poly_multiple = GCD(min_poly, dM);
	min_poly_sqfree = min_poly / min_poly_multiple;
	min_poly_multiple = GCD(min_poly_sqfree, min_poly_multiple);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### squarefree: " << tm.usertime() << endl;
	tm.clear();
	tm.start();
#endif
}

//----------------------------------------------------------//
//----------------------------------------------------------//

void print_poly(const zz_pX& p, std::ostream &ofs){
	ofs << "0";
	for (int i = 0; i <= deg(p); i++)
		ofs << " + "<< coeff(p, i) << "*t^" << i << " ";

}


//----------------------------------------------------------//
//----------------------------------------------------------//
// the component of the lex basis obtained from Xn          //
//----------------------------------------------------------//
//----------------------------------------------------------//
vector<zz_pX>  Block_Sparse_FGLM::get_lex_basis_non_generic(){

	int numvar = n-1;

	PolMatDom PMD(field);
	vector<DenseMatrix<GF>> mat_seq;

	DenseMatrix<GF> mat_seq_left_flat = DenseMatrix<GF>(field, getLength()*M, D);
	DenseMatrix<GF> mat_seq_left_short = DenseMatrix<GF>(field, getGenDeg()*M, D);
#ifdef TIMINGS_ON
	Timer tm;
	tm.clear(); 
	tm.start();
#endif
	// ----------------------------------------------
	// 1. compute the left matrix sequence (U T^i)
	// ----------------------------------------------
	get_matrix_sequence_left(mat_seq_left_flat, numvar);
	for (int i = 0; i < getGenDeg()*M; i++)
		for (int j = 0; j < D; j++)
			mat_seq_left_short.refEntry(i, j) = mat_seq_left_flat.getEntry(i, j);


#ifdef VERBOSE_ON	
	cout << "###OUTPUT### Matrix sequence (U T^i)_i :" << endl;
	cout << "Length d = " << getLength() << endl;
#ifdef EXTRA_VERBOSE_ON	
	cout << "Entries:" << endl;
	for (auto &i: mat_seq_left)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
#endif
#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### left sequence (UT1^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### left sequence (UT1^i): " << tm.realtime() << " (real time)" << endl;
	tm.clear(); 
	tm.start();
#endif

	// ----------------------------------------------
	// 2 get the matrix generator, the minpoly, the vector u_tilde
	// ----------------------------------------------
	PolMatDom::MatrixP u_tilde(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::PMatrix mat_gen;
	zz_pX min_poly_sqfree, min_poly_multiple;
	smith(u_tilde, mat_gen, min_poly_sqfree, min_poly_multiple, mat_seq_left_flat);

	//---------------------------------------
	// 4 cleanup
	//---------------------------------------
	zz_pX n1, n1_inv;
	MatrixDomain<GF> MD2{field};
	DenseMatrix<GF> rhs_2(field, D, 1);
	DenseMatrix<GF> tmp(field, D, 1);
	DenseMatrix<GF> rhs(field, D, 1);

	rhs.refEntry(0, 0) = 1;
	for (int i = 1; i < D; i++)
		rhs.refEntry(i, 0) = 0;
	Omega(n1, u_tilde, mat_gen, mat_seq_left_short, rhs);

	vector <GF::Element> rand_check;
	for (int i = 0; i < n; i++)
		if (i != numvar){
			GF::Element a;
			field.init(a, rand());
			rand_check.emplace_back(a);
		}
	for (long i = 0; i < D; i++){
		rhs.refEntry(i, 0) = 0;
		for (int j = 0; j < n; j++)
			if (j != numvar){
				GF::Element e;
				field.mul(e, rand_check[j], mul_mats[j].getEntry(i, 0));
				field.add(e, e, rhs.getEntry(i, 0));
				rhs.refEntry(i, 0) = e;
			}
	}
	zz_pX n_check;
	Omega(n_check, u_tilde, mat_gen, mat_seq_left_short, rhs);

	for (long i = 0; i < D; i++)
		rhs_2.refEntry(i ,0) = 0;
	for (int j = 0; j < n; j++)
		if (j != numvar){
			MD2.mul(tmp, mul_mats[j], rhs);
			for (long i = 0; i < D; i++){
				GF::Element e;
				field.mul(e, rand_check[j], tmp.getEntry(i, 0));
				field.add(e, e, rhs_2.getEntry(i, 0));
				rhs_2.refEntry(i, 0) = e;
			}
		}
	
	zz_pX n_check2;
	Omega(n_check2, u_tilde, mat_gen, mat_seq_left_short, rhs_2);

	min_poly_sqfree /= min_poly_multiple;
	min_poly_sqfree = GCD(n1*n_check2 - n_check*n_check, min_poly_sqfree);
	InvMod(n1_inv, n1 % min_poly_sqfree, min_poly_sqfree);

	vector<zz_pX> output;

	for (int j = 0; j < n; j++){
		for (int i = 0; i < D; i++)
			rhs.refEntry(i, 0) = mul_mats[j].getEntry(i, 0);
		zz_pX num;
		Omega(num, u_tilde, mat_gen, mat_seq_left_short, rhs);
		output.emplace_back( (num * n1_inv) % min_poly_sqfree );
	}

	output.emplace_back(min_poly_sqfree);

#ifdef OUTPUT_FUNC
	ofs << "p = " << prime << endl;
	ofs << "k = GF(p)\n";
	ofs << "coefs = [";
	for (int i = 0; i < n-1; i++)
		ofs << rand_comb[i] << ", ";
	ofs << rand_comb[n-1];
	ofs << "]\n";
	ofs << "U.<t> = PolynomialRing(k)\n";
	ofs << "R = []" << endl;
	for (int j = 0; j < n; j++){
		ofs << "R.append(";
		print_poly(output[j], ofs);
		ofs <<")"<< endl;
	}
	ofs << "P = ";
	print_poly(output[n], ofs);
	ofs << endl;

	ofs << "Pmult = ";
	print_poly(min_poly_multiple, ofs);
	ofs << endl;

	ofs << "Q.<tt> = U.quo(P)\n";
	for (int i = 0; i < n; i++)
		ofs << "S" << i << " = Q(R[" << i << "])\n";
	ofs << endl;
	ofs << "load (\"" << filename.substr(0, filename.size()-4) << ".sage\")\n";
	ofs << "print (eval(";
	for (int i = 0; i < n-1; i++)
		ofs << "S" << i << ", ";
	ofs << "S" << (n-1) << "))\n";
#endif

	return output; 
}


//----------------------------------------------------------//
//----------------------------------------------------------//
// lex basis obtained from generic var                      //
//----------------------------------------------------------//
//----------------------------------------------------------//
vector<zz_pX>  Block_Sparse_FGLM::get_lex_basis_generic(){

	int numvar = n;

	PolMatDom PMD(field);
	vector<DenseMatrix<GF>> mat_seq;

	DenseMatrix<GF> mat_seq_left_flat = DenseMatrix<GF>(field, getLength()*M, D);
	DenseMatrix<GF> mat_seq_left_short = DenseMatrix<GF>(field, getGenDeg()*M, D);
#ifdef TIMINGS_ON
	Timer tm;
	tm.clear(); 
	tm.start();
#endif
	// ----------------------------------------------
	// 1. compute the left matrix sequence (U T^i)
	// ----------------------------------------------
	get_matrix_sequence_left(mat_seq_left_flat, numvar);
	for (int i = 0; i < getGenDeg()*M; i++)
		for (int j = 0; j < D; j++)
			mat_seq_left_short.refEntry(i, j) = mat_seq_left_flat.getEntry(i, j);


#ifdef VERBOSE_ON	
	cout << "###OUTPUT### Matrix sequence (U T^i)_i :" << endl;
	cout << "Length d = " << getLength() << endl;
#ifdef EXTRA_VERBOSE_ON	
	cout << "Entries:" << endl;
	for (auto &i: mat_seq_left)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
#endif
#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### left sequence (UT1^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### left sequence (UT1^i): " << tm.realtime() << " (real time)" << endl;
	tm.clear(); 
	tm.start();
#endif

	// ----------------------------------------------
	// 2 get the matrix generator, the minpoly, the vector u_tilde
	// ----------------------------------------------
	PolMatDom::MatrixP u_tilde(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::PMatrix mat_gen;
	zz_pX min_poly_sqfree, min_poly_multiple;
	smith(u_tilde, mat_gen, min_poly_sqfree, min_poly_multiple, mat_seq_left_flat);

	//---------------------------------------
	// 3 finding denominator and numerators
	//---------------------------------------
	zz_pX n1, n1_inv;
	DenseMatrix<GF> rhs(field, D, 1);

	rhs.refEntry(0, 0) = 1;
	for (int i = 1; i < D; i++)
		rhs.refEntry(i, 0) = 0;
	Omega(n1, u_tilde, mat_gen, mat_seq_left_short, rhs);
	InvMod(n1_inv, n1 % min_poly_sqfree, min_poly_sqfree);

	vector<zz_pX> output;

	for (int j = 0; j < n; j++){
		for (int i = 0; i < D; i++)
			rhs.refEntry(i, 0) = mul_mats[j].getEntry(i, 0);
		zz_pX num;
		Omega(num, u_tilde, mat_gen, mat_seq_left_short, rhs);
		output.emplace_back( (num * n1_inv) % min_poly_sqfree );
	}

	output.emplace_back(min_poly_sqfree);


#ifdef OUTPUT_FUNC
	ofs << "p = " << prime << endl;
	ofs << "k = GF(p)\n";
	ofs << "coefs = [";
	for (int i = 0; i < n-1; i++)
		ofs << rand_comb[i] << ", ";
	ofs << rand_comb[n-1];
	ofs << "]\n";
	ofs << "U.<t> = PolynomialRing(k)\n";
	ofs << "R = []" << endl;
	for (int j = 0; j < n; j++){
		ofs << "R.append(";
		print_poly(output[j], ofs);
		ofs <<")"<< endl;
	}
	ofs << "P = ";
	print_poly(output[n], ofs);
	ofs << endl;

	ofs << "Pmult = ";
	print_poly(min_poly_multiple, ofs);
	ofs << endl;

	ofs << "Q.<tt> = U.quo(P)\n";
	for (int i = 0; i < n; i++)
		ofs << "S" << i << " = Q(R[" << i << "])\n";
	ofs << endl;
	ofs << "load (\"" << filename.substr(0, filename.size()-4) << ".sage\")\n";
	ofs << "print (eval(";
	for (int i = 0; i < n-1; i++)
		ofs << "S" << i << ", ";
	ofs << "S" << (n-1) << "))\n";
#endif

	return output; 
}


template <typename PolMat>
bool test_order( const PolMat &approx, const PolMat &series, const size_t order )
{
	PolynomialMatrixMulDomain<typename PolMat::Field> PMD(approx.field());
	PolMat prod( approx.field(), approx.rowdim(), series.coldim(), approx.size()+series.size() );
	PMD.mul( prod, approx, series );

	bool test = true;
	size_t d = 0;
	while ( test && d<order )
	{
		for ( size_t i=0; i<prod.rowdim(); ++i )
			for ( size_t j=0; j<prod.coldim(); ++j )
			{
				if ( prod.ref(i,j,d) != 0 )
				{
					test = false;
//					std::cout << d << "\t" << i << "\t" << j << std::endl;
				}
			}
		++d;
	}
	return test;
}

template <typename PolMat>
bool test_kernel( const PolMat &kerbas, const PolMat &pmat )
{
	PolynomialMatrixMulDomain<typename PolMat::Field> PMD(kerbas.field());
	PolMat prod( kerbas.field(), kerbas.rowdim(), pmat.coldim(), kerbas.size()+pmat.size()-1 );
	PMD.mul( prod, kerbas, pmat );

	bool test = true;
	size_t d = 0;
	while ( test && d<kerbas.size()+pmat.size()-1 )
	{
		for ( size_t i=0; i<prod.rowdim(); ++i )
			for ( size_t j=0; j<prod.coldim(); ++j )
			{
				if ( prod.ref(i,j,d) != 0 )
				{
					test = false;
					cout << "degree " << d << endl;
				}
			}
		++d;
	}
	return test;
}

int main_polmatdom( int argc, char **argv ){
	// default arguments
	size_t p = 23068673;  // size of the base field
	size_t M = 4;   // row dimension for the blocks
	size_t D = 512; // vector space dimension / dimension of multiplication matrices
	size_t n = 2;  // number of variables / of multiplication matrices
	size_t threshold = 16;  // threshold MBasis / PMBasis

	static Argument args[] = {
		{ 'p', "-p p", "Set cardinality of the base field to p.", TYPE_INT, &p },
		{ 'n', "-n n", "Set number of variables to n.", TYPE_INT, &n },
		{ 'M', "-M M", "Set the row block dimension to M.", TYPE_INT,       &M },
		{ 'D', "-D D", "Set dimension of test matrices to MxN.", TYPE_INT,  &D },
		{ 't', "-t threshold", "Set threshold mbasis / pmbasis to t.", TYPE_INT,  &threshold },
		END_OF_ARGUMENTS
	};

	parseArguments (argc, argv, args);

	GF field(p);
	long seed = time(NULL);
	typename GF::RandIter rd(field,0,seed);
	PolMatDom PMD( field );
	zz_p::init( p );

#ifdef TEST_APPROX
	cout << "~~~~~~~~~~~STARTING TESTS APPROXIMANTS~~~~~~~~~~~~~" << endl;
	size_t order = 2*ceil(D/(double)M);
	cout << "order for approximant basis : " << order << endl;
	cout << "dimensions of input series matrix : " << 2*M << " x " << M << endl;
	PolMatDom::PMatrix series( field, 2*M, M, order );
	for ( size_t d=0; d<order; ++d )
		for ( size_t i=0; i<2*M; ++i )
			for ( size_t j=0; j<M; ++j )
				rd.random( series.ref( i, j, d ) );
	const vector<int> shift( 2*M, 0 );
	Timer tm;
	{
		cout << "~~~NOW TESTING OLD MBASIS~~~" << endl;
		PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
		tm.clear(); tm.start();
		vector<int> rdeg1 = PMD.old_mbasis( app_bas, series, order, shift );
		tm.stop();
#ifdef VERBOSE_ON
		cout << "###OUTPUT### degrees in approx basis:" << endl;
		PMD.print_degree_matrix( app_bas );
#endif
		cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
		cout << "###TIME### approx basis: " << tm.usertime() << endl;
	}
	{
		cout << "~~~NOW TESTING OLD PMBASIS~~~" << endl;
		PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
		tm.clear(); tm.start();
		vector<int> rdeg2 = PMD.old_pmbasis( app_bas, series, order, shift, threshold );
		tm.stop();
#ifdef VERBOSE_ON
		cout << "###OUTPUT### degrees in approx basis:" << endl;
		PMD.print_degree_matrix( app_bas );
#endif
		cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
		cout << "###TIME### approx basis: " << tm.usertime() << endl;
	}
	{
		cout << "~~~NOW TESTING NEW MBASIS~~~" << endl;
		PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
		tm.clear(); tm.start();
		vector<size_t> mindeg3 = PMD.mbasis( app_bas, series, order, shift );
		tm.stop();
#ifdef VERBOSE_ON
		cout << "###OUTPUT### degrees in approx basis:" << endl;
		PMD.print_degree_matrix( app_bas );
#endif
		cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
		cout << "###TIME### approx basis: " << tm.usertime() << endl;
	}
	{
		cout << "~~~NOW TESTING NEW PMBASIS~~~" << endl;
		PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
		tm.clear(); tm.start();
		vector<size_t> mindeg4 = PMD.pmbasis( app_bas, series, order, shift, threshold );
		tm.stop();
#ifdef VERBOSE_ON
		cout << "###OUTPUT### degrees in approx basis:" << endl;
		PMD.print_degree_matrix( app_bas );
#endif
		cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
		cout << "###TIME### approx basis: " << tm.usertime() << endl;
	}
	{
		cout << "~~~NOW TESTING POPOV_PMBASIS~~~" << endl;
		PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
		tm.clear(); tm.start();
		vector<size_t> mindeg5 = PMD.popov_pmbasis( app_bas, series, order, shift, threshold );
		tm.stop();
#ifdef VERBOSE_ON
		cout << "###OUTPUT### degrees in approx basis: " << endl;
		PMD.print_degree_matrix( app_bas );
#endif
		cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
		cout << "###TIME### approx basis: " << tm.usertime() << endl;
	}
#endif
#ifdef TEST_KERNEL
	{
		cout << "~~~NOW TESTING KERNEL BASIS~~~" << endl;
		size_t degree = ceil(D/(double)M); // should be close to the degree of the MatrixBM output
		cout << "dimensions of input matrix : " << M << " x " << M-1 << endl;
		cout << "degree of input matrix : " << degree << endl;
		PolMatDom::PMatrix pmat( field, M, M-1, degree+1 );
		for ( size_t d=0; d<=degree; ++d )
			for ( size_t i=0; i<M; ++i )
				for ( size_t j=0; j<M-1; ++j )
					rd.random( pmat.ref( i, j, d ) );
		Timer tm;
		PolMatDom::PMatrix kerbas( field, 1, M, 0 ); // one is enough except extreme bad luck
		tm.clear(); tm.start();
		PMD.kernel_basis( kerbas, pmat, threshold );
		tm.stop();
		cout << "###OUTPUT### degrees in kernel basis: " << endl;
		PMD.print_degree_matrix( kerbas );
		cout << "###CORRECTNESS### is kernel: " << test_kernel( kerbas, pmat ) << endl;
		cout << "###TIME### kernel basis: " << tm.usertime() << endl;
	}
#endif // TEST_KERNEL
#ifdef TEST_INVARIANT_FACTOR
	{
		cout << "~~~NOW TESTING INVARIANT FACTOR~~~" << endl;
		size_t degree = ceil(D/(double)M); // should be close to the degree of the MatrixBM output
		cout << "dimensions of input matrix : " << M << " x " << M << endl;
		cout << "input matrix random of degree " << degree << endl;
		PolMatDom::PMatrix pmat( field, M, M, degree+1 );
		for ( size_t d=0; d<=degree; ++d )
			for ( size_t i=0; i<M; ++i )
				for ( size_t j=0; j<M; ++j )
					rd.random( pmat.ref( i, j, d ) );
		Timer tm;
		vector<zz_pX> left_multiplier( M );
		zz_pX factor;
		cout << "OK" << endl;
		tm.clear(); tm.start();
		PMD.largest_invariant_factor( left_multiplier, factor, pmat, threshold );
		tm.stop();
		//cout << "###OUTPUT### degrees in kernel basis: " << endl;
		//PMD.print_degree_matrix( kerbas );
		//cout << "###CORRECTNESS### is kernel: " << test_kernel( kerbas, pmat ) << endl;
		cout << "###TIME### kernel basis: " << tm.usertime() << endl;
	}
#endif // TEST_INVARIANT_FACTOR
	return 0;
}

int main_fglm( int argc, char **argv ){
	// default arguments
	size_t M = 4;   // row dimension for the blocks
	string F = "";
	size_t threshold = 16;  // threshold MBasis / PMBasis

	static Argument args[] = {
		{ 'M', "-M M", "Set the row block dimension to M.", TYPE_INT,       &M },
		{ 't', "-t threshold", "Set threshold mbasis / pmbasis to t.", TYPE_INT,  &threshold },
		{ 'F', "-F F", "Read input from file F", TYPE_STR,  &F },
		END_OF_ARGUMENTS
	};	

	parseArguments (argc, argv, args);

	cout << "blocking dimension:" << " M=" << M << endl;
	cout << "threshold mbasis/pmbasis (advice: take 128 if p==9001): " << threshold << endl;
	cout << "F=" << F << endl;
	if (F == "")
		return 0;
	
	InputMatrices mat(F);
	Block_Sparse_FGLM l(M, mat, threshold);
	l.get_lex_basis_non_generic();
	l.get_lex_basis_generic();

	return 0;
}
