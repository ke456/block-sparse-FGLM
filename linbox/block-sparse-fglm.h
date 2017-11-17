/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */
#ifndef BLOCK_SPARSE_FGLM_H
#define BLOCK_SPARSE_FGLM_H

#include <string>

// #include <linbox/integer.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/matrix/permutation-matrix.h>
#include "linbox/matrix/polynomial-matrix.h"
#include "linbox/algorithms/polynomial-matrix/polynomial-matrix-domain.h"
#include "fflas-ffpack/fflas-ffpack.h"
#include <NTL/lzz_pX.h>

// Givaro polynomials
#include <givaro/givpoly1.h>
#include "linbox/ring/givaro-poly.h"

typedef Givaro::Modular<double> GF;

//-----------------------------------------------//
//-----------------------------------------------//
//        Polynomial matrix stuff                //
//-----------------------------------------------//
//-----------------------------------------------//
class PolMatDom {

    public:
	typedef std::vector<typename GF::Element> Polynomial;
	typedef LinBox::PolynomialMatrix<LinBox::PMType::polfirst,LinBox::PMStorage::plain,GF> MatrixP;
	typedef LinBox::PolynomialMatrix<LinBox::PMType::matfirst,LinBox::PMStorage::plain,GF> PMatrix;
	typedef Givaro::Poly1Dom<GF>::Element GivPoly;
	
    private:
	const GF* _field;
	//Givaro::Poly1Dom<GF> _PD;
	LinBox::BlasMatrixDomain<GF> _BMD;
	LinBox::PolynomialMatrixMulDomain<GF> _PMMD;
	
    public:
	
	PolMatDom(const GF &f) :
		_field(&f),
		//_PD(f),
		_BMD(f),
		_PMMD(f) { }

	inline const GF& field() const {return *_field;}

	template<typename PolMat>
	void print_degree_matrix( const PolMat &pmat ) const;

	void xgcd( const Polynomial & a, const Polynomial & b, Polynomial & g, Polynomial & u, Polynomial & v );
	void divide( const Polynomial & a, const Polynomial & b, Polynomial & q );

	//void slow_mul( PMatrix & prod, const PMatrix & mat1, const PMatrix & mat2 )

	// Smith form of a nonsingular matrix; also computes the unimodular factors
	void SmithForm( std::vector<Polynomial> &smith, PMatrix &lfac, PMatrix &rfac, const PMatrix &pmat, const size_t threshold=16 );

	// mbasis algorithm to compute approximant bases
	// ideally, all these should be const, but issues because of Linbox's multiplication of polmats
	std::vector<int> old_mbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				     const std::vector<int> &shift=std::vector<int>() );
	std::vector<size_t> mbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				    const std::vector<int> &shift=std::vector<int>(), bool resUpdate=false );

	// pmbasis divide and conquer algorithm to compute approximant bases
	std::vector<int> old_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				      const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );
	std::vector<size_t> pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				     const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );
	std::vector<size_t> popov_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
					   const std::vector<int> &shift=std::vector<int>(), const size_t threshold=16 );

	// computing s-owP kernel basis
	void kernel_basis( PMatrix & kerbas, const PMatrix & pmat, const size_t threshold=16 );

	// Matrix Berlekamp-Massey: returns a matrix generator for a sequence of matrices
	template<typename Matrix>
	void MatrixBerlekampMassey( PMatrix &mat_gen, PMatrix &mat_num, const std::vector<Matrix> & mat_seq, const size_t threshold=16 );
	
};


//-----------------------------------------------//
//-----------------------------------------------//
//        Functions for polynomial matrices      //
//-----------------------------------------------//
//-----------------------------------------------//

//shifts every entry by d
void shift(PolMatDom::PMatrix &result, const PolMatDom::PMatrix &mat, int row, int col, int deg);
void mat_to_poly (PolMatDom::Polynomial &p, PolMatDom::MatrixP &mat, int size);
void mat_resize (GF &field, PolMatDom::MatrixP &mat, int size);


//-----------------------------------------------//
//-----------------------------------------------//
//           Container for input data            //
//-----------------------------------------------//
//-----------------------------------------------//
struct InputMatrices{
public:
	int n, D, p;
	std::vector<std::vector<int>> x;
	std::vector<std::vector<int>> y;
	std::vector<std::vector<double>> data;
	std::vector<double> sparsity;
	std::string name; // name of the system
	std::string filename;
	InputMatrices(std::string & filename);

};

//-----------------------------------------------//
//-----------------------------------------------//
//        The main class for FGLM                //
//-----------------------------------------------//
//-----------------------------------------------//
class Block_Sparse_FGLM{
	// the current field
	GF field;
	int prime;

	int D; // vector space dimension / dimension of multiplication matrices
	int M; // number of blocks (set to number of CPUs?)
	size_t n; // number of variables and size of vector mul_mats
	size_t threshold; // FIXME temporary: threshold MBasis/PMBasis
	
	// stores the multiplication matrices T_i
	std::vector<LinBox::SparseMatrix<GF>> mul_mats;
	LinBox::DenseMatrix<GF> V; //right side of U*T1*V
	std::vector<LinBox::DenseMatrix<GF>> mat_seq_left; // store U*T1^i
	// coeffs in the random combination
	std::vector<GF::Element> rand_comb;
	// sparsities
	std::vector<double> sparsity;
	std::string name;
	std::string filename;


	std::ofstream ofs;

	/* Helpers                                           */
	template<typename Matrix>
	void create_random_matrix(Matrix &m);
	
	// Computes sequence (U T1^i)
	void get_matrix_sequence_left(std::vector<LinBox::DenseMatrix<GF>>&, int numvar);
	
	// Computes sequence (UT1^i)V
	void get_matrix_sequence(std::vector<LinBox::DenseMatrix<GF>> &,
	                         std::vector<LinBox::DenseMatrix<GF>> &,
	                         LinBox::DenseMatrix<GF> &,
				 int,
	                         size_t);


    public:
	// length of the sequence:
	size_t getLength() const { return 2*ceil(D/(double)M); };
	// generic degree in matrix generator:
	size_t getGenDeg() const { return ceil(D/(double)M); };
	size_t getThreshold() const { return threshold; }; // FIXME temporary: threshold MBasis/PMBasis
	
	/* CTOR                                              */
	Block_Sparse_FGLM(size_t M, InputMatrices& mat, size_t threshold);

	std::vector<NTL::zz_pX> find_lex_basis();
	std::vector<NTL::zz_pX> find_lex_basis(const std::vector<LinBox::DenseMatrix<GF>> &, int numvar);
};



#endif
