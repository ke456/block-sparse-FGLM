#ifndef BLOCK_SPARSE_FGLM_H
#define BLOCK_SPARSE_FGLM_H

#include <linbox/integer.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/matrix/permutation-matrix.h>
#include "linbox/matrix/polynomial-matrix.h"
#include "linbox/algorithms/polynomial-matrix/polynomial-matrix-domain.h"
#include "fflas-ffpack/fflas-ffpack.h"


typedef Givaro::Modular<double> GF;
class Block_Sparse_FGLM{
	// the current field
	GF field;

	int D; // vector space dimension / dimension of multiplication matrices
	int M; // number of blocks (set to number of CPUs?)
	size_t n; // number of variables and size of vector mul_mats

	// length of the sequence:
	size_t getLength() const { return 2*ceil(D/(double)M); };
	// generic degree in matrix generator:
	size_t getGenDeg() const { return ceil(D/(double)M); };
	
	// stores the multiplication matrices T_i
	std::vector<LinBox::SparseMatrix<GF>> mul_mats;

	LinBox::DenseMatrix<GF> V; //right side of U*T1*V
	std::vector<LinBox::DenseMatrix<GF>> mat_seq_left; // store U*T1^i

	/* Helpers                                           */
	template<typename Matrix>
	void create_random_matrix(Matrix &m);
	
	// Computes sequence (U T1^i)
	void get_matrix_sequence_left(std::vector<LinBox::DenseMatrix<GF>> &);
	
	// Computes sequence (UT1^i)V
	void get_matrix_sequence(std::vector<LinBox::DenseMatrix<GF>> &,
	                         std::vector<LinBox::DenseMatrix<GF>> &,
	                         LinBox::DenseMatrix<GF> &,
													 int,
	                         size_t);


	public:
	/* CTOR                                              */
	Block_Sparse_FGLM(const GF &, int, int, size_t);
	Block_Sparse_FGLM(const GF &field, int D, int M, size_t n, std::string& s);

	void find_lex_basis();
};

class PolMatDom {

	public:

	typedef std::vector<typename GF::Element> Polynomial;
	typedef LinBox::PolynomialMatrix<LinBox::PMType::polfirst,LinBox::PMStorage::plain,GF> MatrixP;
	typedef LinBox::PolynomialMatrix<LinBox::PMType::matfirst,LinBox::PMStorage::plain,GF> PMatrix;

	private:

	const GF* _field;
	LinBox::PolynomialMatrixMulDomain<GF> _PMMD;
	LinBox::BlasMatrixDomain<GF> _BMD;

	public:

	PolMatDom(const GF &f) : _field(&f), _PMMD(f), _BMD(f) {	}

	inline const GF& field() const {return *_field;}

	template<typename PolMat>
	void print_degree_matrix( const PolMat &pmat ) const;

	void xgcd( const Polynomial & a, const Polynomial & b, Polynomial & g, Polynomial & u, Polynomial & v );
	void divide( const Polynomial & a, const Polynomial & b, Polynomial & q );

	// Smith form of a nonsingular matrix; also computes the unimodular factors
	void SmithForm( std::vector<Polynomial> &smith, MatrixP &lfac, MatrixP &rfac, const MatrixP &pmat );

	// mbasis algorithm to compute approximant bases
	// ideally, all these should be const, but issues because of Linbox's multiplication of polmats
	std::vector<int> old_mbasis( PMatrix &approx, const PMatrix &series, const size_t order, const std::vector<int> &shift=std::vector<int>() );
	std::vector<size_t> mbasis( PMatrix &approx, const PMatrix &series, const size_t order, const std::vector<int> &shift=std::vector<int>(), bool resUpdate=false );

	// pmbasis divide and conquer algorithm to compute approximant bases
	std::vector<int> old_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );
	std::vector<size_t> pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );
	std::vector<size_t> popov_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );


	// Matrix Berlekamp-Massey: returns a matrix generator for a sequence of matrices
	template<typename Matrix>
	void MatrixBerlekampMassey( MatrixP &mat_gen, MatrixP &mat_num, const std::vector<Matrix> & mat_seq );

};

#endif
