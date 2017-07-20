#define TIMINGS_ON // comment out if having timings is not relevant
//#define EXTRA_VERBOSE_ON // extra detailed printed objects, like multiplication matrix and polynomial matrices... unreadable except for very small dimensions
#define VERBOSE_ON // some objects printed for testing purposes, but not the biggest ones (large constant matrices, polynomial matrices..)
//#define NAIVE_ON
#define EXTRA_VERBOSE_ON
#define WARNINGS_ON // comment out if having warnings for heuristic parts is irrelevant --> should probably be 'on'
#define SPARSITY_COUNT // shows the sparsity of the matrices
#include "block-sparse-fglm.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace LinBox;
using namespace std;

// reads matrices from s
Block_Sparse_FGLM::Block_Sparse_FGLM(const GF &field, int D, int M, size_t n, string& s):
  field(field), 
  D(D), 
  M(M), 
  n(n), 
  mul_mats(n, SparseMatrix<GF>(field,D,D)), 
  V(field,D,M), 
  mat_seq_left(2*ceil(D/(double)M), DenseMatrix<GF>(field,M,D)) 
{
  string line;
  ifstream file;
  file.open (s);
  getline(file, line);
  getline(file, line);
  getline(file, line);

	create_random_matrix(V);
 
  vector<double> sparsity_count;
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
      sparsity_count.emplace_back((double)(count)/max_entries);
      count = 0;
    }
    else{
      count++;
      GF::Element a(a_int);
      mul_mats[index].refEntry(i,j) = a;
    }
  }
  file.close();
  
#ifdef SPARSITY_COUNT
  cout << "sparsity: ";
  for (auto i: sparsity_count)
    cout << i << " ";
  cout << endl;
#endif
#ifdef EXTRA_VERBOSE_ON
  for (auto &i : mul_mats)
    i.write(cout << "mul mat", Tag::FileFormat::Maple)<<endl;
#endif
}


Block_Sparse_FGLM::Block_Sparse_FGLM(const GF &field, int D, int M, size_t n): 
  field(field), 
  D(D), 
  M(M), 
  n(n), 
  mul_mats(n, SparseMatrix<GF>(field,D,D)), 
  V(field,D,M),
  mat_seq_left(2*ceil(D/(double)M),DenseMatrix<GF>(field,M,D)) 
{
	for ( size_t k=0; k<n; ++k )
		create_random_matrix(mul_mats[k]);
	create_random_matrix(V);
	srand(time(NULL));
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

void Block_Sparse_FGLM::get_matrix_sequence_left(vector<DenseMatrix<GF>> &v){
	MatrixDomain<GF> MD{field};

	// initialize the left block (random MxD)
	vector<DenseMatrix<GF>> U_rows(M, DenseMatrix<GF>(field,1,D));
	for (auto &i : U_rows){
		create_random_matrix(i);
#ifdef EXTRA_VERBOSE_ON
		i.write(cout << "###OUTPUT### U_row: ",Tag::FileFormat::Maple)<<endl;
#endif
	}
	
	// stores U_i*T1^j at mat_seq[i][j]
	vector<vector<DenseMatrix<GF>>> mat_seq(M);
	for (auto &i : mat_seq){
		i = vector<DenseMatrix<GF>>(this->getLength(),DenseMatrix<GF>(field,1,D));
	}

	// initialize the first multiplication matrix (random DxD)
	auto &T1 = mul_mats[0];
#ifdef EXTRA_VERBOSE_ON
	T1.write(cout << "###OUTPUT### Multiplication matrix T1:"<<endl, Tag::FileFormat::Maple)<<endl;
#endif

	// compute sequence in a parallel fashion
#pragma omp parallel for num_threads(M)
	for (int i  = 0; i < M; i++){
		MatrixDomain<GF> MD2{field};
		vector<DenseMatrix<GF>> temp_mat_seq(this->getLength(), DenseMatrix<GF>(field,1,D)); 
		temp_mat_seq[0] = U_rows[i];
		for (size_t j  = 1; j < this->getLength(); j++){
			MD2.mul(temp_mat_seq[j],temp_mat_seq[j-1],T1);
		}
		mat_seq[i] = temp_mat_seq;
	}
	
	for (size_t i = 0; i < this->getLength(); i++){
		auto &temp = v[i];
		for (int row = 0; row < M; row++){
			for (int col = 0; col < D; col++){
				GF::Element a;
				mat_seq[row][i].getEntry(a,0,col);
				temp.refEntry(row,col) = a;
			}
		}
	} 
}

void Block_Sparse_FGLM::get_matrix_sequence
(vector<DenseMatrix<GF>> &v, 
 vector<DenseMatrix<GF>> &l, 
 DenseMatrix<GF> &V,
 size_t to){
	// gather all the matrices of v in a single (seq_length*M) x D matrix
	MatrixDomain<GF> MD(field);
	DenseMatrix<GF> mat(field, to*M, D);
	for (size_t i = 0; i < to; i++){
		auto &m = l[i];
		for (int row = 0; row < M; row++){
			int r = i * M + row; // starting point for mat
			for (int col = 0; col < D; col++){
				GF::Element a;
				m.getEntry(a,row,col);
				mat.refEntry(r,col) = a;
			}
		}
	}
	
	// multiplication
	DenseMatrix<GF> result(field, to*M,M);
	MD.mul(result,mat,V);
	
	v.resize(to, DenseMatrix<GF>(field,M,M));
	for (size_t i = 0; i < to; i++){
		v[i] = DenseMatrix<GF>(field, M, M);
		for (int row = 0; row < M; row++){
			int r = i * M + row;
			for (int col = 0; col < M; col++){
				GF::Element a;
				result.getEntry(a,r,col);
				v[i].refEntry(row,col) = a;
			}
		}
	}
}

void Block_Sparse_FGLM::find_lex_basis(){
#ifdef TIMINGS_ON
	auto start = chrono::high_resolution_clock::now();
#endif
	// 1. compute the "left" matrix sequence (U T1^i)
	get_matrix_sequence_left(mat_seq_left);
	
#ifdef VERBOSE_ON	
	cout << "###OUTPUT### Matrix sequence (U T1^i)_i :" << endl;
	cout << "Length d = " << this->getLength() << endl;
#ifdef EXTRA_VERBOSE_ON	
    cout << "Entries:" << endl;
	for (auto &i: mat_seq_left)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
#endif
#ifdef TIMINGS_ON
	auto end = chrono::high_resolution_clock::now();
	cout << "###TIME### left sequence (UT1^i): " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
	start = chrono::high_resolution_clock::now();
#endif
	vector<DenseMatrix<GF>> mat_seq(getLength(), DenseMatrix<GF>(field,M,M));
	// 2. compute the total matrix sequence (UT1^i)V
	get_matrix_sequence(mat_seq, mat_seq_left, V, getLength());
#ifdef TIMINGS_ON
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### sequence (UT1^i)V: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
	start = chrono::high_resolution_clock::now();
#endif
#ifdef VERBOSE_ON
	cout << "###OUTPUT### Matrix sequence (U T1^i V)_i :" << endl;
	cout << "Length d = " << this->getLength() << endl;
	cout << "Generic generator degree: deg = " << this->getGenDeg() << endl;
#ifdef EXTRA_VERBOSE_ON
	cout << "Entries:" << endl;
	for (auto &i: mat_seq)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
#endif
	// 3. compute generator and numerator in Matrix Berlekamp Massey: reversed sequence = mat_gen/mat_num
	// note: * generator is in Popov form
	//       * degree of row i of numerator < degree of row i of generator
	PolMatDom PMD( field );
	PolMatDom::MatrixP mat_gen(PMD.field(),M,M,this->getLength());
	PolMatDom::MatrixP mat_num(PMD.field(),M,M,this->getLength());
	PMD.MatrixBerlekampMassey<DenseMatrix<GF>>( mat_gen, mat_num, mat_seq );
#ifdef TIMINGS_ON
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### Matrix Berlekamp Massey: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
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
	vector<PolMatDom::Polynomial> smith( M );
	PolMatDom::MatrixP lfac(PMD.field(),M,M,M*this->getLength()+1);
	PolMatDom::MatrixP rfac(PMD.field(),M,M,M*this->getLength()+1);
#ifdef TIMINGS_ON
	start = chrono::high_resolution_clock::now();
#endif
	PMD.SmithForm( smith, lfac, rfac, mat_gen );
#ifdef TIMINGS_ON
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### Smith form and transformations: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
#endif
#ifdef NAIVE_ON
	DenseMatrix<GF> U(field,M,D);
	MatrixDomain<GF> MD(field);
	vector<DenseMatrix<GF>> lst(this->getLength(), DenseMatrix<GF>(field,M,D));
	auto &T1 = mul_mats[0];
	create_random_matrix(U);
	lst[0] = U;
	start = chrono::high_resolution_clock::now();
	for (unsigned int i = 1; i < this->getLength(); i++){
		MD.mul(lst[i],lst[i-1],T1);
		lst[i] = U;
	}
#ifdef EXTRA_VERBOSE_ON
	for (auto &i : lst)
		i.write(cout<<"U: ", Tag::FileFormat::Maple)<<endl;
#endif
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### sequence (UT1^i) naive: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl;
#endif
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

vector<int> PolMatDom::mbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const vector<int> &shift ) const
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

vector<int> PolMatDom::pmbasis( PolMatDom::PMatrix &approx, const PolMatDom::PMatrix &series, const size_t order, const std::vector<int> &shift, const size_t threshold )
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
		std::vector<int> rdeg = mbasis( approx, series, order, shift );
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
			rdeg = pmbasis( approx1, res1, order1, rdeg, threshold ); // first recursive call
		} // end of scope: res1 is deallocated here
		{
			PolMatDom::PMatrix res2( series.field(), m, n, order2 ); // second residual: midproduct 
			this->_PMMD.midproductgen( res2, approx1, series, true, order1+1, order1+order2 ); // res2 = (approx1*series / X^order1) mod X^order2
			//midproduct( res2, approx1, series );
			rdeg = pmbasis( approx2, res2, order2, rdeg, threshold ); // second recursive call
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

template <typename PolMat>
void PolMatDom::midproduct( PolMat &res, const PolMat &approx, const PolMat &series )
{

}


void PolMatDom::SmithForm( vector<PolMatDom::Polynomial> &smith, PolMatDom::MatrixP &lfac, MatrixP &rfac, const PolMatDom::MatrixP &pmat ) const {
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
	vector<int> rdeg = this->mbasis( app_bas, series, order, shift );
	// Note: appbas size is likely order+1, although we know here no entry of degree order (we could reduce this size)
#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### First approximant basis: shifted row degrees and matrix degrees:" << endl;
	cout << rdeg << endl;
	this->print_degree_matrix(app_bas);
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif

	// extract the left factor lfac which is the bottom left block,
	// as well as Transpose(LeftHermite(pmat)) which is the transpose of the bottom right block
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
	vector<int> rdeg2 = this->mbasis( app_bas2, series2, order, shift );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### Second approximant basis: shifted row degrees and matrix degrees:" << endl;
	cout << rdeg2 << endl;
	this->print_degree_matrix(app_bas2);
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas2 << endl;
#endif
#endif

	// extract the right factor rfac, which is the transpose of the bottom left block
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
void PolMatDom::MatrixBerlekampMassey( PolMatDom::MatrixP &mat_gen, PolMatDom::MatrixP &mat_num, const vector<Matrix> & mat_seq ) const {
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

	// 2. compute approximant basis in reduced form
	vector<int> rdeg = this->mbasis( app_bas, series, d, shift );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Approximant basis: output rdeg and basis degrees" << endl;
	cout << rdeg << endl;
	this->print_degree_matrix( app_bas );
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif
#ifdef WARNINGS_ON
	// so far we are assuming the degrees are evenly balanced
	// --> easily retrieve the Popov approximant basis
	// This should hold in almost all cases (unless the field size is too small)
	// This assumption could be avoided easily but implies to rewrite OrderBasis::M_Basis
	bool test = true;
	for ( size_t i=0; i<2*M-1; ++i )
		if ( rdeg[i] != rdeg[i+1] )
			test = false;
	if (not test) {
		cout << "~~~WARNING(MatrixBM)~~~ unexpected degrees in approximant basis" << endl;
		cout << "   ------->>> rest of the computation may give wrong results." << endl;
	}
#endif

	// 3. Transform app_bas into Popov form
	// Right now, assuming the row degree 'rdeg' is constant
	// --> suffices to left-multiply by invert of leading matrix

	// retrieve inverse of leading matrix
	Matrix lmat = app_bas[rdeg[0]];
#ifdef EXTRA_VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### leading matrix of reduced approximant basis:" << endl;
	cout << lmat << endl;
#endif
	this->_BMD.invin( lmat ); // lmat is now the inverse of leading_matrix(app_bas)

	// create Popov approximant basis and fill it
 	for ( size_t k=0; k<app_bas.size(); ++k ) {
		this->_BMD.mulin_right( lmat, app_bas[k] );
	}

#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Popov approximant basis degrees:" << endl;
	this->print_degree_matrix(app_bas);
#ifdef EXTRA_VERBOSE_ON
	cout << "Basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif

	// 4. copy into mat_gen and mat_num
	mat_gen.setsize( rdeg[0]+1 );
	mat_num.setsize( rdeg[0]+1 );
	for ( size_t i=0; i<M; ++i )
	for ( size_t j=0; j<M; ++j )
	for ( int k=0; k<=rdeg[0]; ++k )
	{
		mat_gen.ref(i,j,k) = app_bas.get(i,j,k);
		mat_num.ref(i,j,k) = app_bas.get(i,j+M,k);
	}
}

int main( int argc, char **argv ){
	// default arguments
	size_t p = 23068673;  // size of the base field
	size_t M = 4;   // row dimension for the blocks
	size_t D = 512; // vector space dimension / dimension of multiplication matrices
	size_t n = 2;  // number of variables / of multiplication matrices
	string s = "";

	static Argument args[] = {
		{ 'p', "-p p", "Set cardinality of the base field to p.", TYPE_INT, &p },
		{ 'n', "-n n", "Set number of variables to n.", TYPE_INT, &n },
		{ 'M', "-M M", "Set the row block dimension to M.", TYPE_INT,       &M },
		{ 'D', "-D D", "Set dimension of test matrices to MxN.", TYPE_INT,  &D },
		{ 'F', "-F F", "Read input from file F", TYPE_STR,  &s },
		END_OF_ARGUMENTS
	};	

	parseArguments (argc, argv, args);

#ifdef WARNINGS_ON
	if ( D%M != 0 ) {
		cout << "~~~WARNING~~~ block dimension M does not divide vector space dimension D" << endl;
		cout << "     ----->>> results of approximant basis / Smith form unpredictable." << endl;
	}
#endif

	cout << "s=" << s<< endl;
	if (s == ""){
	  GF field(p);
	  Block_Sparse_FGLM l(field, D, M, n);
	  l.find_lex_basis();
	}
	else{
	  // we read the first lines here 
	  string line;
	  ifstream file;
	  file.open (s);
	  getline(file, line);
	  p = stoi(line);
	  getline(file, line);
	  n = stoi(line);
	  getline(file, line);
	  D = stoi(line);
	  cout << "read file " << s << " with p=" << p << " n=" << n << " D=" << D << endl;
	  file.close();
	  GF field(p);
	  Block_Sparse_FGLM l(field, D, M, n, s);
	  l.find_lex_basis();
	  return 0;
	}

}			
