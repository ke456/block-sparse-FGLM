#include "block-sparse-fglm.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <ctime>
#include <cstdlib>

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

void PolMatDom::xgcd( const Polynomial & a, const Polynomial & b, Polynomial & g, Polynomial & u, Polynomial & v )
{
  const size_t deg = max(a.size(),b.size());
  const size_t order = 1 + 2*deg;
  vector<int> shift = { 0, 0, (int)deg };
	PolMatDom::PMatrix series( this->field(), 3, 1, order );
  for ( size_t d=0; d<a.size(); ++d )
    series.ref(0,0,d) = a[d];
  for ( size_t d=0; d<b.size(); ++d )
    series.ref(1,0,d) = b[d];
  series.ref(2,0,0) = this->field().mOne;

  PolMatDom::PMatrix approx( this->field(), 3, 3, order-1 );
  pmbasis( approx, series, order, shift );

  g = approx(2,2);
  u = approx(2,0);
  v = approx(2,1);
}

void PolMatDom::divide( const Polynomial & a, const Polynomial & b, Polynomial & q )
{
  const size_t deg = max(a.size(),b.size());
  const size_t order = 1+deg;
  vector<int> shift = { 0, (int)deg };
	PolMatDom::PMatrix series( this->field(), 2, 1, order );
  for ( size_t d=0; d<b.size(); ++d )
    series.ref(0,0,d) = -b[d];
  for ( size_t d=0; d<a.size(); ++d )
    series.ref(1,0,d) = a[d];

  PolMatDom::PMatrix approx( this->field(), 2, 2, order-1 );
  pmbasis( approx, series, order, shift );

  this->print_degree_matrix( approx );

  q = approx(1,0);
}

vector<size_t> PolMatDom::mbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const vector<int> &shift, bool resUpdate )
{
	/** Algorithm M-Basis-One as detailed in Section 3 of
	 *  [Jeannerod, Neiger, Villard. Fast Computation of approximant bases in
	 *  canonical form. Preprint, 2017]
	 **/
	/** Input:
	 *   - approx: m x m square polynomial matrix, approximation basis
	 *   - series: m x n polynomial matrix of size = order+1, series to approximate
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
	 *   - Compute and store in 'approx' a shifted minimal approximation basis for (series,order,shift)
	 **/
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

void PolMatDom::SmithForm( vector<PolMatDom::Polynomial> &smith, PolMatDom::MatrixP &lfac, MatrixP &rfac, const PolMatDom::MatrixP &pmat ) {
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
	vector<size_t> mindeg = this->pmbasis( app_bas, series, order, shift );
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
	vector<size_t> mindeg2 = this->popov_pmbasis( app_bas2, series2, order, shift );
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

template<typename Matrix>
void PolMatDom::MatrixBerlekampMassey( PolMatDom::MatrixP &mat_gen, PolMatDom::MatrixP &mat_num, const vector<Matrix> & mat_seq ) {
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
	vector<size_t> mindeg = this->pmbasis( app_bas, series, d, shift );
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

template <typename PolMat>
bool test_order( const PolMat &approx, const PolMat &series, const size_t order )
{
	PolynomialMatrixMulDomain<typename PolMat::Field> PMD(approx.field());
	PolMat prod( approx.field(), approx.rowdim(), series.coldim(), approx.size()+series.size() );
	PMD.mul( prod, approx, series );

	//std::cout << prod << std::endl;
	
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