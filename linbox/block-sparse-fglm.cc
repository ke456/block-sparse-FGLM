/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */

#define TIMINGS_ON // to activate timings
#define SPARSITY_COUNT // shows the sparsity of the matrices
// #define OUTPUT_FUNC // outputs the computed functions

#include "block-sparse-fglm.h"

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

//----------------------------------------------------------//
// pretty print an NTL polynomial                           //
//----------------------------------------------------------//
void print_poly(const zz_pX& p, std::ostream &ofs){
	ofs << "0";
	for (int i = 0; i <= deg(p); i++)
		ofs << " + "<< coeff(p, i) << "*t^" << i << " ";

}

//-----------------------------------------------//
//-----------------------------------------------//
//              Polynomial matrix                //
//-----------------------------------------------//
//-----------------------------------------------//

//shifts every entry by d
void shift(PolMatDom::PMatrix &result, const PolMatDom::PMatrix &mat, int deg){
	for (int i = 0; i < mat.rowdim(); i++)
		for (int j = 0; j < mat.coldim(); j++){
			for (int d = 0; d < deg && (d+deg) < mat(i,j).size(); d++){
				auto element = mat.get(i, j, d+deg);
				result.ref(i, j, d) = element;
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
	PolMatDom::MatrixP temp(field, mat.rowdim(), mat.coldim(), size);
	for (int j = 0; j < mat.rowdim(); j++)
		for (int k = 0; k < mat.coldim(); k++)
			for (int i = 0; i < size; i++)
				temp.ref(j, k, i) = mat.ref(j, k , i);
	mat = temp;
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
	approx.resize(0); // to put zeroes everywhere.. there might be a better way to do it but it seems LinBox's approx.clear() fails
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
		size_t rank = FFPACK::PLUQ( res_const.field(), FFLAS::FflasNonUnit,
					    m, n, res_const.getWritePointer(), res_const.getStride(),
					    P.getWritePointer(), Q.getWritePointer() );

		// compute a part of the left kernel basis of res_const:
		// -Lbot Ltop^-1 , stored in Lbot
		// Note: the full kernel basis is [ -Lbot Ltop^-1 | I ] P
		View Ltop( res_const, 0, 0, rank, rank ); // top part of lower triangular matrix in PLUQ
		View Lbot( res_const, rank, 0, m-rank, rank ); // bottom part of lower triangular matrix in PLUQ
		FFLAS::ftrsm( approx.field(), FFLAS::FflasRight, FFLAS::FflasLower,
			      FFLAS::FflasNoTrans, FFLAS::FflasUnit,
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

		// NOTE: resUpdate = True may not be supported currently; assuming False
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

vector<size_t> PolMatDom::pmbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const std::vector<int> &shift )
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

	if ( order <= this->getPMBasisThreshold() )
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
			mindeg = pmbasis( approx1, res1, order1, shift ); // first recursive call
		} // end of scope: res1 is deallocated here
		{
			vector<int> rdeg( shift ); // shifted row degrees = mindeg + shift
			for ( size_t i=0; i<m; ++i )
				rdeg[i] += mindeg[i];
			PolMatDom::PMatrix res2( series.field(), m, n, order2 ); // second residual: midproduct 
			this->_PMMD.midproductgen( res2, approx1, series, true, order1+1, order1+order2 ); // res2 = (approx1*series / X^order1) mod X^order2
			vector<size_t> mindeg2( m );
			mindeg2 = pmbasis( approx2, res2, order2, rdeg ); // second recursive call
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

/*! \brief Computing a kernel basis of a polynomial matrix
 *
 *  Given an m x n polynomial matrix F, and a shift s = (s1,...,sn),
 *  this function computes a left kernel basis of F in ordered
 *  weak Popov form (for the uniform shift).
 *
 *  Algorithm based on PM-Basis at a high order
 *  --> faster algorithms are known in some cases,
 *  cf. Zhou-Labahn-Storjohann ISSAC 2012
 *
 * \param kerbas: storage for the output kernel basis
 * \param pmat: input polynomial matrix
 * \return the shifted minimal degree of the kernel basis (or the shifted pivot degree and index?)
 */
/* Not: for simplicity, specific to the present situation -> in general, it might return only a part of the kernel basis */
void PolMatDom::kernel_basis( PMatrix & kerbas, const PMatrix & pmat )
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
	vector<size_t> mindeg = this->pmbasis( appbas, series, order, shift );
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

void PolMatDom::largest_invariant_factor( vector<zz_pX> & left_multiplier, zz_pX & factor, const PMatrix & pmat, const size_t position )
{
	// Recall from .h : it is assumed that left_multiplier has been initialized with m zero polynomials
	size_t m = pmat.rowdim();

	// 1. main computation: find vector in the kernel of all columns except one
	//     (being careful with the case of an 1x1 matrix pmat..
	if ( m > 1 )
	{
		PolMatDom::PMatrix subcols( this->field(), m, m-1, pmat.size() );
		for (size_t d = 0; d < pmat.size(); ++d)
		for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < m; ++j)
		{
			if ( j < position )
				subcols.ref(i,j,d) = pmat.get(i,j,d);
			else if ( j > position )
				subcols.ref(i,j-1,d) = pmat.get(i,j,d);
		}

		PolMatDom::PMatrix kerbas( this->field(), 1, m, 0 );
		this->kernel_basis( kerbas, subcols );
		// copy this into left_multiplier
		for ( size_t i=0; i<m; ++i )
		{
			for (size_t d=0; d<kerbas.size(); ++d)
				SetCoeff(left_multiplier[i], d, (long)kerbas.get(0,i,d));
			left_multiplier[i].normalize();
		}
	}
	else // pmat is an 1x1 matrix
	{
		left_multiplier[0] = 1;
		left_multiplier[0].normalize();
	}

	// 2. compute factor
	factor = 0;
	zz_pX pol(pmat.size());
	for ( size_t i=0; i<m; ++i )
	{
		for (size_t d=0; d<pmat.size(); ++d)
			SetCoeff(pol, d, (long)pmat.get(i,position,d));
		factor += left_multiplier[i] * pol;
	}

	// 3. make sure the factor is monic
	factor.normalize();
	zz_p lc = LeadCoeff( factor );
	factor /= lc;
	for ( size_t i=0; i<m; ++i )
		left_multiplier[i] /= lc;
}

template<typename Matrix>
void PolMatDom::MatrixBerlekampMassey( PolMatDom::PMatrix &mat_gen, PolMatDom::PMatrix &mat_num, const vector<Matrix> & mat_seq ) {
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

	// 2. compute approximant basis in ordered weak Popov form
	vector<size_t> mindeg = this->pmbasis( app_bas, series, d, shift );

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
	U_rows(M, DenseMatrix<GF>(field, 1, D)),
	V(field, D, M),
	// mul_mats(n+1, SparseMatrix<GF>(field, D, D)),
	Emul_mats(n+1, Eigen::SparseMatrix<double,Eigen::RowMajor>(D, D)),
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

	for (int i = 0; i < n; i++){
// DOING EIGEN MATRICES
		vector<Eigen::Triplet<double>> coefficients;

		for (int j = 0; j < mat.x[i].size(); j++){
			int xx = mat.x[i][j];
			int yy = mat.y[i][j];
			// GF::Element a(mat.data[i][j]);
			// mul_mats[i].refEntry(xx, yy) = a;
			// GF::Element e;
			// field.mul(e, rand_comb[i], a);
			// field.add(e, e, mul_mats[n].refEntry(xx, yy));
			// mul_mats[n].refEntry(xx, yy) = e;
			coefficients.push_back(Eigen::Triplet<double> (xx, yy, mat.data[i][j]));

		}
		Emul_mats[i].setFromTriplets(coefficients.begin(), coefficients.end());
		Emul_mats[i].makeCompressed();

		Emul_mats[n] = Emul_mats[n] + (double) rand_comb[i] * Emul_mats[i];
		Emul_mats[n].makeCompressed();
	}


	long nbM = 0;
	for (int k = 0; k < Emul_mats[n].outerSize(); ++k)
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Emul_mats[n], k); it; ++it){
			long coef = (long) it.value() % prime ;
			if (coef != 0)
				nbM++;
			it.valueRef() = coef;
		}

			// it.valueRef() =  (long) it.value() % prime ;


	
#ifdef SPARSITY_COUNT
	cout << "sparsity: ";
	for (auto i: sparsity)
		cout << i << " ";
	cout << endl;
  cout << "D: " << D << endl;
  cout << "sparsity M: " << ((double)nbM)/(D*D) << endl;

#endif

	zz_p::init(prime);

	create_random_matrix(V);
	for (auto &i : U_rows){
		create_random_matrix(i);
	}
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



void Block_Sparse_FGLM::get_matrix_sequence_left(DenseMatrix<GF> &v_flat, int numvar, int number){
	MatrixDomain<GF> MD{field};
	v_flat = DenseMatrix<GF>(field, number*M, D);

	Eigen::setNbThreads(1);

#pragma omp parallel for num_threads(M)
	for (int i  = 0; i < M; i++){

		Eigen::Matrix<double, 1, Eigen::Dynamic> row(D), row2(D);
		for (long j = 0; j < D; j++)
			row(0, j) = (double) (long) U_rows[i].getEntry(0, j);
		for (size_t k  = 0; k < number; k++) {
			for (long j = 0; j < D; j++)
				v_flat.refEntry(k*M + i, j) = (row(0, j) = ((long)(row(0, j)) % prime));
			row2 = row * Emul_mats[numvar];
			row = row2;
		}
	}

}



//-------------------------------------------------------//
// Computes sequence v = (UT1^i)V                        //
// assumes that v_flat holds the matrix of all (U T1^i)  //
//-------------------------------------------------------//
void Block_Sparse_FGLM::get_matrix_sequence(vector<DenseMatrix<GF>> &v, 
					    const DenseMatrix<GF> &l_flat, 
					    const DenseMatrix<GF> &V){

	// multiplication
	MatrixDomain<GF> MD(field);
	int to = l_flat.rowdim() / M;
	int c = V.coldim(); 
	DenseMatrix<GF> result(field, to*M, c);
	MD.mul(result, l_flat, V);

	v = vector<DenseMatrix<GF>>(to, DenseMatrix<GF>(field, M, c));
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
// reconstructs a number x number matrix of numerators   //
//-------------------------------------------------------//
void Block_Sparse_FGLM::Omega(vector<zz_pX> & numerator, const PolMatDom::MatrixP &u_tilde, const PolMatDom::PMatrix &mat_gen,
			      const vector<DenseMatrix<GF>> &seq,
			      int number_row, int number_col){

	PolMatDom PMD(field);
	PolynomialMatrixMulDomain<GF> PMMD(field);
	int deg = seq.size();

	PolMatDom::PMatrix polys(PMD.field(), M, number_col, deg);
	// creating the poly matrix from the sequence in reverse order
	for (int k = 0; k < number_col; k++){
		int index = 0;
		for (int j = seq.size()-1; j >= 0; j--){
			for (int q = 0; q < M; q++){
				auto element = seq[j].getEntry(q, k);
				polys.ref(q, k, index) = element;
			}
			index++;
		}
	}

	PolMatDom::PMatrix N(PMD.field(), M, number_col, 1+2*deg); //was :1+getLength()
	PolMatDom::PMatrix N_shift(PMD.field(), M, number_col, 1+2*deg);//was :1+getLength()
	for (int k = 0; k < number_col; k++)
		for (long j = 0; j < M; j++)
			for (long i = 0; i < 1+2*deg; i++){ //was :1+getLength()
				N.ref(j, k, i) = 0;
				N_shift.ref(j, k, i) = 0;
			}
	
	PMMD.mul(N, mat_gen, polys);
	shift(N_shift, N, deg);

// we could do this product using only number_row rows
	PolMatDom::MatrixP n_mat(PMD.field(), u_tilde.rowdim(), number_col, D+1);
	for (int k = 0; k < number_col; k++)
		for (long j = 0; j < u_tilde.rowdim(); j++)
			for (long i = 0; i < (D+1); i++)
				n_mat.ref(j, k, i) = 0;
	PMMD.mul(n_mat, u_tilde, N_shift);
	mat_resize(field, n_mat, D);

	numerator.resize(number_row*number_col);
	long index = 0;
	for (long j = 0; j < number_row; j++){
		for (int k = 0; k < number_col; k++){
			numerator[index] = 0;
			for (int i  = 0; i < D; i++)
				SetCoeff(numerator[index], i, (long)n_mat.get(j, k, i));
			index++;
		}
	}
}

//--------------------------------------------------------------------------------//
// builds a random V                                                              //
// applies matrix-BM and smith                                                    //
// return u_tilde, the minimal matrix generator and the minpoly                   //
// put "number" rows in u_tilde
//--------------------------------------------------------------------------------//
void Block_Sparse_FGLM::smith(PolMatDom::MatrixP &u_tilde, PolMatDom::PMatrix &mat_gen, zz_pX &min_poly, 
			      const vector<DenseMatrix<GF>> &mat_seq,
			      int number){

#ifdef TIMINGS_ON
	Timer tm;
#endif

	//---------------------------------------
	// 1. matrix Berlekamp-Massey
	//---------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	int length = mat_seq.size();
	PolMatDom PMD(field, getThreshold());
	mat_gen = PolMatDom::PMatrix (PMD.field(), M, M, length);
	PolMatDom::PMatrix mat_num(PMD.field(), M, M, length);
 	PMD.MatrixBerlekampMassey<DenseMatrix<GF>>( mat_gen, mat_num, mat_seq );

	// //---------------------------------------
	// // 2. matrix Berlekamp-Massey
	// //---------------------------------------
	// PolMatDom PMD(field,getThreshold());
	// mat_gen = PolMatDom::PMatrix (PMD.field(), M, M, getLength());
	// PolMatDom::PMatrix mat_num(PMD.field(), M, M, getLength());

#ifdef TIMINGS_ON
	tm.stop();
	cout << " ###TIME### Matrix Berlekamp Massey: " << tm.usertime() << endl; 
#endif

	//---------------------------------------
	// 3. u_tilde and minpoly
	//---------------------------------------
	vector<zz_pX> u_tilde_ntl( M );

	u_tilde = PolMatDom::MatrixP(PMD.field(), number, M, D+3);
	for (long m = 0; m < number; m++){
		PMD.largest_invariant_factor( u_tilde_ntl, min_poly, mat_gen, m );
		for ( size_t i=0; i<M; ++i ){
			for ( long d=0; d<=deg(u_tilde_ntl[i]); ++d ) // not size_t (unsigned)
			{
				size_t coeff_size_t;
				conv( coeff_size_t, coeff( u_tilde_ntl[i], d ) );
				u_tilde.ref(m, i, d) = coeff_size_t;
			}
		}
	}

#ifdef TIMINGS_ON
	tm.stop();
	cout << " ###TIME### Computing u_tilde: " << tm.usertime() << endl;;
#endif

}


//----------------------------------------------------------//
//----------------------------------------------------------//
// the component of the lex basis obtained from Xn          //
//----------------------------------------------------------//
//----------------------------------------------------------//
vector<zz_pX>  Block_Sparse_FGLM::get_lex_basis_non_generic(){

#ifdef TIMINGS_ON
	Timer tm, tm_total;
	cout << "###------------ ENTER NON GENERIC LEX -------------" << tm.usertime() << endl;
	tm_total.clear(); 
	tm_total.start();
#endif

	int numvar = n-1;
	PolMatDom PMD(field);
	vector<DenseMatrix<GF>> mat_seq;
	DenseMatrix<GF> mat_seq_left_flat = DenseMatrix<GF>(field, getLength()*M, D);
	DenseMatrix<GF> mat_seq_left_short = DenseMatrix<GF>(field, getGenDeg()*M, D);

	// ----------------------------------------------
	// 1. compute the left matrix sequence (U T^i)
	// ----------------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	get_matrix_sequence_left(mat_seq_left_flat, numvar, getLength());
	for (int i = 0; i < getGenDeg()*M; i++)
		for (int j = 0; j < D; j++)
			mat_seq_left_short.refEntry(i, j) = mat_seq_left_flat.getEntry(i, j);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### left sequence (UT1^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### left sequence (UT1^i): " << tm.realtime() << " (real time)" << endl;
#endif

	// ----------------------------------------------
	// 2. get the matrix generator, the minpoly, the vector u_tilde
	// ----------------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif
	PolMatDom::MatrixP u_tilde(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::PMatrix mat_gen;
	zz_pX min_poly, min_poly_sqfree, min_poly_multiple;

	get_matrix_sequence(mat_seq, mat_seq_left_flat, V);

	smith(u_tilde, mat_gen, min_poly, mat_seq, M);
	zz_pX dM = diff(min_poly);
	min_poly_multiple = GCD(min_poly, dM);
	min_poly_sqfree = min_poly / min_poly_multiple;
	min_poly_multiple = GCD(min_poly_sqfree, min_poly_multiple);
	min_poly_sqfree /= min_poly_multiple; // only keep roots w/o multiplicities

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Total Smith / u_tilde / min_poly: " << tm.usertime() << endl;
#endif

	//---------------------------------------
	// 3. cleanup
	//---------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	zz_pX n1, n1_inv;
	MatrixDomain<GF> MD2{field};
	DenseMatrix<GF> rhs_2(field, D, 1);
	DenseMatrix<GF> tmp(field, D, 1);
	DenseMatrix<GF> rhs(field, D, 1);
	vector<zz_pX> vec_numers, vec_numers_1, tmp_vec;

	rhs.refEntry(0, 0) = 1;
	for (int i = 1; i < D; i++)
		rhs.refEntry(i, 0) = 0;

	vector<DenseMatrix<GF>> seq(mat_seq_left_short.rowdim() / M, DenseMatrix<GF>(field, M, 1));
	get_matrix_sequence(seq, mat_seq_left_short, rhs);
	Omega(vec_numers_1, u_tilde, mat_gen, seq, M, 1);  
	n1 = vec_numers_1[0];

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
				field.mul(e, rand_check[j], (double) Emul_mats[j].coeff(i, 0));
				field.add(e, e, rhs.getEntry(i, 0));
				rhs.refEntry(i, 0) = e;
			}
	}
	zz_pX n_check;
	get_matrix_sequence(seq, mat_seq_left_short, rhs);
	Omega(tmp_vec, u_tilde, mat_gen, seq);
	n_check = tmp_vec[0];

	for (long i = 0; i < D; i++)
		rhs_2.refEntry(i ,0) = 0;

	
	for (int j = 0; j < n; j++)
		if (j != numvar){

			Eigen::Matrix<double, Eigen::Dynamic, 1> col(D), col2(D);
			for (long jj = 0; jj < D; jj++)
				col(jj, 0) = (double) (long) rhs.getEntry(jj, 0);
			col2 = Emul_mats[j]*col;
			for (long jj = 0; jj < D; jj++)
				tmp.refEntry(jj, 0) =  ((long)(col2(jj, 0)) % prime);

			// MD2.mul(tmp, mul_mats[j], rhs);


			for (long i = 0; i < D; i++){
				GF::Element e;
				field.mul(e, rand_check[j], tmp.getEntry(i, 0));
				field.add(e, e, rhs_2.getEntry(i, 0));
				rhs_2.refEntry(i, 0) = e;
			}
		}
	
	zz_pX n_check2;
	get_matrix_sequence(seq, mat_seq_left_short, rhs_2);
	Omega(tmp_vec, u_tilde, mat_gen, seq);
	n_check2 = tmp_vec[0];
	min_poly_sqfree = GCD(n1*n_check2 - n_check*n_check, min_poly_sqfree);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### clean-up: " << tm.usertime() << endl;
#endif

	//---------------------------------------
	// 4. get numerators -> output
	//---------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	long d = deg(min_poly_sqfree);
	long dp = deg(min_poly);
	if (d == 0)
		n1_inv = 0;
	else
		InvMod(n1_inv, n1 % min_poly_sqfree, min_poly_sqfree);

	vector<zz_pX> output;
	for (int j = 0; j < n; j++){
		for (int i = 0; i < D; i++)
			rhs.refEntry(i, 0) = Emul_mats[j].coeff(i, 0);
		get_matrix_sequence(seq, mat_seq_left_short, rhs);
		Omega(tmp_vec, u_tilde, mat_gen, seq);
		output.emplace_back( (tmp_vec[0] * n1_inv) % min_poly_sqfree );
	}
	output.emplace_back(min_poly_sqfree);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### parametrizations: " << tm.usertime() << endl;
#endif
        // fix this
	if (d == 0){
#ifdef TIMINGS_ON
		tm_total.stop();
		cout << "###TIME### Total real time " << tm_total.realtime() << endl;
		cout << "###TIME### Total user time " << tm_total.usertime() << endl;	
		cout << "###------------ ABORT NON GENERIC LEX -------------" << tm.usertime() << endl;
#endif
		return output;
	}


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

	if (D == d){
#ifdef TIMINGS_ON
		tm_total.stop();
		cout << "###TIME### Total real time " << tm_total.realtime() << endl;
		cout << "###TIME### Total user time " << tm_total.usertime() << endl;	
		cout << "###------------ EXIT NON GENERIC LEX -------------" << tm.usertime() << endl;
#endif
		return output;
	}

	//---------------------------------------
	// 5. prepare the update
	//---------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif
	// +4 extra for safety (?)
	int short_length = (int)(2*ceil((D-d)/(double)M)) + 4;

	zz_pXModulus min_poly_modulus(min_poly_sqfree);
	zz_pX revP, B;
	for (long i = 0; i <= d; i++)
		SetCoeff(revP, d-i, coeff(min_poly_sqfree, i));
	for (long i = 0; i <= dp; i++)
		SetCoeff(B, dp-i, coeff(min_poly, i));
	zz_pX rest = B / revP;
	cout << "revP: " << revP << endl;
	cout << "rest: " << rest << endl;
	zz_pX inv_revP = zz_pX(0);
	if (rest != 1)
		inv_revP = InvMod(revP % rest, rest);
	zz_pX inv_rest = InvMod(rest % revP, revP);

	zz_pX T; // the random linear form, written mod min_poly_sqfree
	for (long i = 0; i < n; i++)
		T += output[i] * (long) rand_comb[i];

        // correction terms for the parametrizations
	vector<mat_zz_p> correction_parametrizations;
	for (long i = 0; i < n+1; i++){
		mat_zz_p mati;
		mati.SetDims(M, short_length/2);
		correction_parametrizations.emplace_back(mati);
	}

	for (long m = 0; m < M; m++){
		zz_pX A1 = vec_numers_1[m];
		zz_pX rA1;
		for (long i = 0; i < dp; i++)
			SetCoeff(rA1, i, coeff(A1, dp-1-i));
		zz_pX num1 = (rA1*inv_rest) % revP;
		zz_pX num2 = (rA1*inv_revP) % rest;
		zz_pX num3 = (rA1-num1*rest-num2*revP) / B;
		if (coeff(min_poly_sqfree, 0) == 0)
			num1 = num1+revP*num3;

		vec_zz_p coeffs;
		zz_pX gen_series = num1*InvTrunc(revP, d);
		coeffs.SetLength(d);
		for (long i = 0; i < d; i++)
			coeffs[i] = coeff(gen_series, i);
		
		vec_zz_p values = ProjectPowers(coeffs, short_length / 2, T, min_poly_modulus);
		for (long i = 0; i < short_length/2; i++)
			correction_parametrizations[0][m][i] = values[i];
		
		for (long j = 0; j < n; j++){
			zz_pXMultiplier mult(output[j], min_poly_modulus);
			vec_zz_p Xi_coeffs = UpdateMap(coeffs, mult, min_poly_modulus);
			values = ProjectPowers(Xi_coeffs, short_length / 2, T, min_poly_modulus);
			for (long i = 0; i < short_length/2; i++)
				correction_parametrizations[j+1][m][i] = values[i];
		}
	}

	// correction terms for BM
	seq = vector<DenseMatrix<GF>>(mat_seq_left_short.rowdim() / M, DenseMatrix<GF>(field, M, V.coldim()));
	get_matrix_sequence(seq, mat_seq_left_short, V);
	Omega(vec_numers, u_tilde, mat_gen, seq, M, M);

	LinBox::DenseMatrix<GF> carry_over(field, short_length*M, M);

	for (long k = 0; k < M; k++){
		for (long ell = 0; ell < M; ell++){
			zz_pX A1 = vec_numers[k*M + ell];
			zz_pX rA1;
			for (long i = 0; i < deg(min_poly); i++)
				SetCoeff(rA1, i, coeff(A1, deg(min_poly)-1-i));
			zz_pX num1 = (rA1*inv_rest) % revP;
			zz_pX num2 = (rA1*inv_revP) % rest;
			zz_pX num3 = (rA1-num1*rest-num2*revP) / B;
			if (coeff(min_poly_sqfree, 0) == 0)
				num1 = num1+revP*num3;

			vec_zz_p coeffs;
			zz_pX gen_series = num1*InvTrunc(revP, d);
			coeffs.SetLength(d);
			for (long i = 0; i < d; i++)
				coeffs[i] = coeff(gen_series, i);
		
			vec_zz_p sequence =  ProjectPowers(coeffs, short_length, T, min_poly_modulus);
			for (long i = 0; i < short_length; i++){
				carry_over.refEntry(i*M + k, ell) = sequence[i]._zz_p__rep;
			}
		}
	}


#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### get correction terms: " << tm.usertime() << endl;
#endif

	// ----------------------------------------------
	// 6. compute the left matrix sequence (U T^i), generic T
	// ----------------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	DenseMatrix<GF> mat_seq_left_generic_flat = DenseMatrix<GF>(field, short_length*M, D);
	DenseMatrix<GF> mat_seq_left_generic_short = DenseMatrix<GF>(field, (short_length / 2)*M, D);
	
	get_matrix_sequence_left(mat_seq_left_generic_flat, n, short_length);
	for (int i = 0; i < (short_length / 2)*M; i++)
		for (int j = 0; j < D; j++)
			mat_seq_left_generic_short.refEntry(i, j) = mat_seq_left_generic_flat.getEntry(i, j);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### second left sequence (UT^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### second left sequence (UT^i): " << tm.realtime() << " (real time)" << endl;
#endif

	// ----------------------------------------------
	// 7. get the new matrix generator, minpoly, u_tilde
	// ----------------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	PolMatDom::MatrixP u_tilde_generic(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::PMatrix mat_gen_generic;
	zz_pX min_poly_generic, min_poly_sqfree_generic, min_poly_multiple_generic;

	vector<DenseMatrix<GF>> mat_seq_generic;
	get_matrix_sequence(mat_seq_generic, mat_seq_left_generic_flat, V);
	for (long i = 0; i < mat_seq_left_generic_flat.rowdim(); i++)
		for (long j = 0; j < M; j++){
			auto e = mat_seq_generic[i / M].getEntry(i % M, j);
			field.sub(e, e, carry_over.getEntry(i, j));
			mat_seq_generic[i / M].refEntry(i % M, j) = e;
		}

	smith(u_tilde_generic, mat_gen_generic, min_poly_generic, mat_seq_generic);
	zz_pX dM_generic = diff(min_poly_generic);
	min_poly_multiple_generic = GCD(min_poly_generic, dM_generic);
	min_poly_sqfree_generic = min_poly_generic / min_poly_multiple_generic;
	min_poly_multiple_generic = GCD(min_poly_sqfree_generic, min_poly_multiple_generic);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Total second Smith / u_tilde / min_poly: " << tm.usertime() << endl;
#endif

	//---------------------------------------
	// 8. finding denominator and numerators
	//---------------------------------------

#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	rhs.refEntry(0, 0) = 1;
	for (int i = 1; i < D; i++)
		rhs.refEntry(i, 0) = 0;

	seq = vector<DenseMatrix<GF>>(mat_seq_left_generic_short.rowdim() / M, DenseMatrix<GF>(field, M, 1));
	get_matrix_sequence(seq, mat_seq_left_generic_short, rhs);

	for (long m = 0; m < M; m++)
		for (long i = 0; i < short_length/2; i++){
			auto e = seq[i].getEntry(m, 0);
			field.sub(e, e, correction_parametrizations[0][m][i]._zz_p__rep);
			seq[i].refEntry(m, 0) = e;
		}
	Omega(tmp_vec, u_tilde_generic, mat_gen_generic, seq);
	n1 = tmp_vec[0];
	InvMod(n1_inv, n1 % min_poly_sqfree_generic, min_poly_sqfree_generic);

	vector<zz_pX> output_generic;
	for (int j = 0; j < n; j++){
		for (int i = 0; i < D; i++)
			rhs.refEntry(i, 0) = Emul_mats[j].coeff(i, 0);
		vector<zz_pX> tmp_vec;
		get_matrix_sequence(seq, mat_seq_left_generic_short, rhs);

		for (long m = 0; m < M; m++)
			for (long i = 0; i < short_length/2; i++){
				auto e = seq[i].getEntry(m, 0);
				field.sub(e, e, correction_parametrizations[j+1][m][i]._zz_p__rep);
				seq[i].refEntry(m, 0) = e;
			}

		Omega(tmp_vec, u_tilde_generic, mat_gen_generic, seq);
		output_generic.emplace_back( (tmp_vec[0] * n1_inv) % min_poly_sqfree_generic );
	}

	output_generic.emplace_back(min_poly_sqfree_generic);



#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### second parametrizations: " << tm.usertime() << endl;
#endif

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
		print_poly(output_generic[j], ofs);
		ofs <<")"<< endl;
	}
	ofs << "P = ";
	print_poly(output_generic[n], ofs);
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

#ifdef TIMINGS_ON
	tm_total.stop();
	cout << "###TIME### Total real time " << tm_total.realtime() << endl;
	cout << "###TIME### Total user time " << tm_total.usertime() << endl;	
	cout << "###------------ EXIT NON GENERIC LEX -------------" << tm.usertime() << endl;
#endif

	return output; 
}


//----------------------------------------------------------//
//----------------------------------------------------------//
// lex basis obtained from generic var                      //
//----------------------------------------------------------//
//----------------------------------------------------------//
vector<zz_pX>  Block_Sparse_FGLM::get_lex_basis_generic(){

#ifdef TIMINGS_ON
	Timer tm, tm_total;
	cout << "###------------ ENTER GENERIC LEX -------------" << tm.usertime() << endl;
	tm_total.clear(); 
	tm_total.start();
#endif

	int numvar = n;
	PolMatDom PMD(field);
	DenseMatrix<GF> mat_seq_left_flat = DenseMatrix<GF>(field, getLength()*M, D);
	DenseMatrix<GF> mat_seq_left_short = DenseMatrix<GF>(field, getGenDeg()*M, D);

	// ----------------------------------------------
	// 1. compute the left matrix sequence (U T^i)
	// ----------------------------------------------
#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	get_matrix_sequence_left(mat_seq_left_flat, numvar, getLength());
	for (int i = 0; i < getGenDeg()*M; i++)
		for (int j = 0; j < D; j++)
			mat_seq_left_short.refEntry(i, j) = mat_seq_left_flat.getEntry(i, j);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### left sequence (UT^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### left sequence (UT^i): " << tm.realtime() << " (real time)" << endl;
#endif

	// ----------------------------------------------
	// 2 get the matrix generator, the minpoly, the vector u_tilde
	// ----------------------------------------------
#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	PolMatDom::MatrixP u_tilde(PMD.field(), 1, M, M*getLength()+1);
	PolMatDom::PMatrix mat_gen;
	zz_pX min_poly, min_poly_sqfree, min_poly_multiple;

	vector<DenseMatrix<GF>> mat_seq;
	get_matrix_sequence(mat_seq, mat_seq_left_flat, V);
	smith(u_tilde, mat_gen, min_poly, mat_seq);
//	smith(u_tilde, mat_gen, min_poly, mat_seq_left_flat);
	zz_pX dM = diff(min_poly);
	min_poly_multiple = GCD(min_poly, dM);
	min_poly_sqfree = min_poly / min_poly_multiple;
	min_poly_multiple = GCD(min_poly_sqfree, min_poly_multiple);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###DEGREE## min_poly deg: " << deg(min_poly) << endl;
	cout << "###TIME### Total Smith / u_tilde / min_poly: " << tm.usertime() << endl;
#endif

	//---------------------------------------
	// 3 finding denominator and numerators
	//---------------------------------------
#ifdef TIMINGS_ON
	tm.clear(); 
	tm.start();
#endif

	zz_pX n1, n1_inv;
	DenseMatrix<GF> rhs(field, D, 1);
	vector<zz_pX> tmp_vec;

	rhs.refEntry(0, 0) = 1;
	for (int i = 1; i < D; i++)
		rhs.refEntry(i, 0) = 0;

	vector<DenseMatrix<GF>> seq(mat_seq_left_short.rowdim() / M, DenseMatrix<GF>(field, M, rhs.coldim()));

	get_matrix_sequence(seq, mat_seq_left_short, rhs);
	Omega(tmp_vec, u_tilde, mat_gen, seq);  
	n1 = tmp_vec[0];
	InvMod(n1_inv, n1 % min_poly_sqfree, min_poly_sqfree);

	vector<zz_pX> output;

	for (int j = 0; j < n; j++){
		for (int i = 0; i < D; i++)
			rhs.refEntry(i, 0) = Emul_mats[j].coeff(i, 0);
		vector<zz_pX> tmp_vec;
		get_matrix_sequence(seq, mat_seq_left_short, rhs);
		Omega(tmp_vec, u_tilde, mat_gen, seq);  
		output.emplace_back( (tmp_vec[0] * n1_inv) % min_poly_sqfree );
	}

	output.emplace_back(min_poly_sqfree);

#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### parametrizations: " << tm.usertime() << endl;
#endif

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

#ifdef TIMINGS_ON
	tm_total.stop();
	cout << "###TIME### Total real time " << tm_total.realtime() << endl;
	cout << "###TIME### Total user time " << tm_total.usertime() << endl;	
	cout << "###------------ EXIT GENERIC LEX -------------" << tm.usertime() << endl;
#endif

	return output; 
}

int main( int argc, char **argv ){
	// default arguments
	size_t M = 4;   // row dimension for the blocks
	string F = "";
	size_t threshold = 128;  // threshold MBasis / PMBasis

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
