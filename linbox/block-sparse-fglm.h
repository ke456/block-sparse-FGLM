#ifndef BLOCK_SPARSE_FGLM_H
#define BLOCK_SPARSE_FGLM_H

#include <linbox/integer.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/algorithms/polynomial-matrix/order-basis.h>

typedef Givaro::Modular<int> GF;

class Block_Sparse_FGLM{
	// stores the multiplication matrices T_i
	std::vector<LinBox::SparseMatrix<GF>> mul_mats;
	// the current field
	GF field;

	int D; // vector space dimension / dimension of multiplication matrices
	int M; // number of blocks (set to number of CPUs?)
	
	LinBox::DenseMatrix<GF> V; //right side of U*T1*V
	// Vincent: we store V and not U?

	/* Helpers                                           */
	template<typename Matrix>
	void create_random_matrix(Matrix &m);
	
	// Computes sequence (U T1^i)
	void get_matrix_sequence_left(std::vector<LinBox::DenseMatrix<GF>> &);
	
	// Computes sequence (UT1^i)V
	void get_matrix_sequence(std::vector<LinBox::DenseMatrix<GF>> &);

	public:
	/* CTOR                                              */
	Block_Sparse_FGLM(int,int,const GF &);

	void find_lex_basis();
};

class PolMatDom {

	public:

	typedef std::vector<typename GF::Element> Polynomial;
	//typedef PolynomialMatrix<PMType::polfirst,PMStorage::plain,GF> MatrixP;
	typedef LinBox::PolynomialMatrix<LinBox::PMType::matfirst,LinBox::PMStorage::plain,GF> PMatrix;

	private:

	const GF* _field;
	LinBox::PolynomialMatrixMulDomain<GF> _PMMD;
	LinBox::BlasMatrixDomain<GF> _BMD;

	public:

	PolMatDom(const GF &f) : _field(&f), _PMMD(f), _BMD(f) {	}

	inline const GF& field() const {return *_field;}

	// Smith form of a nonsingular matrix; also computes the unimodular factors
	size_t SmithForm( std::vector<Polynomial> &smith, PMatrix &lfac, PMatrix &rfac, const PMatrix &pmat ) const;

	// Matrix Berlekamp-Massey: returns a matrix generator for a sequence of matrices
	template<typename Matrix>
	size_t MatrixBerlekampMassey( PMatrix &matgen, std::vector<Matrix> seq ) const;

	void print_pmat( const PMatrix &pmat ) const;

};

#endif
