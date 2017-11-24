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
	LinBox::BlasMatrixDomain<GF> _BMD;
	LinBox::PolynomialMatrixMulDomain<GF> _PMMD;
	size_t _pmbasis_threshold;
	
    public:
	
	PolMatDom(const GF &f) :
		_field(&f),
		_BMD(f),
		_PMMD(f),
		_pmbasis_threshold(16)
	{ }

	PolMatDom(const GF &f, const size_t threshold) :
		_field(&f),
		_BMD(f),
		_PMMD(f),
		_pmbasis_threshold(threshold)
	{ }

	inline const GF& field() const {return *_field;}

	inline const size_t getPMBasisThreshold() const { return _pmbasis_threshold; }

	template<typename PolMat>
	void print_degree_matrix( const PolMat &pmat ) const;

	// mbasis algorithm to compute approximant bases
	// ideally, all these should be const, but issues because of Linbox's multiplication of polmats
	std::vector<int> old_mbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				     const std::vector<int> &shift=std::vector<int>() );
	std::vector<size_t> mbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				    const std::vector<int> &shift=std::vector<int>(), bool resUpdate=false );

	// pmbasis divide and conquer algorithm to compute approximant bases
	std::vector<int> old_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				      const std::vector<int> &shift=std::vector<int>() );
	std::vector<size_t> pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
				     const std::vector<int> &shift=std::vector<int>() );
	std::vector<size_t> popov_pmbasis( PMatrix &approx, const PMatrix &series, const size_t order, 
					   const std::vector<int> &shift=std::vector<int>() );

	// computing s-owP kernel basis
	void kernel_basis( PMatrix & kerbas, const PMatrix & pmat );

	// (Heuristic) computes a vector v(x) and a polynomial f(x) such that
	// v(x) pmat(x) = [0...0 f(x) 0 ... 0], where f(x) is the largest Smith factor,
	// and it is at position "position"
	// Assumes 'position' is an integer between 0 and pmat.rowdim()-1
	// Assumes left_multiplier has been initialized with pmat.rowdim() empty polynomials
	void largest_invariant_factor( std::vector<NTL::zz_pX> &left_multiplier, NTL::zz_pX &factor, const PMatrix &pmat, const size_t position );

	// (Heuristic) Smith form of a nonsingular matrix; also computes the unimodular factors
	void SmithForm( std::vector<Polynomial> &smith, PMatrix &lfac, PMatrix &rfac, const PMatrix &pmat );


	// Matrix Berlekamp-Massey: returns a matrix generator for a sequence of matrices
	template<typename Matrix>
	void MatrixBerlekampMassey( PMatrix &mat_gen, PMatrix &mat_num, const std::vector<Matrix> & mat_seq );
	
};


//-----------------------------------------------//
//-----------------------------------------------//
//        Functions for polynomial matrices      //
//-----------------------------------------------//
//-----------------------------------------------//

//shifts every entry by deg
void shift(PolMatDom::PMatrix &result, const PolMatDom::PMatrix &mat, int deg);
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

	std::vector<LinBox::DenseMatrix<GF>> U_rows;
	LinBox::DenseMatrix<GF> V; //right side of U*T1*V
	// stores the multiplication matrices T_i
	std::vector<LinBox::SparseMatrix<GF>> mul_mats;
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
	
        //-------------------------------------------------------//
        // puts U*mul_mats[numvar]^i into v, i=0...2D/M          //
        // v_flat is v flattened into a single matrix            //
        // numvar = n means we are using a random combination    //
        //-------------------------------------------------------//
	void get_matrix_sequence_left(LinBox::DenseMatrix<GF>& v_flat,
				      int numvar, int number);
	
        //-------------------------------------------------------//
	// Computes sequence (UT1^i)V                            //
        // assumes that v_flat holds the matrix of all (U T1^i)  //
        //-------------------------------------------------------//
	void get_matrix_sequence(std::vector<LinBox::DenseMatrix<GF>> & result,
				 const LinBox::DenseMatrix<GF> & v_flat,
	                         const LinBox::DenseMatrix<GF> & V);

        //-------------------------------------------------------//
        //-------------------------------------------------------//
	void Omega(std::vector<NTL::zz_pX> & numerator, const PolMatDom::MatrixP &u_tilde,
		   const PolMatDom::PMatrix &mat_gen,
		   const std::vector<LinBox::DenseMatrix<GF>> &seq,
		   int number_row = 1, int number_col = 1
		);

       //--------------------------------------------------------------------------------//
       // builds a random V                                                              //
       // applies matrix-BM and smith                                                    //
       // return u_tilde, the minimal matrix generator and the minpoly                   //
       //--------------------------------------------------------------------------------//
	void smith(PolMatDom::MatrixP &u_tilde, PolMatDom::PMatrix &mat_gen, NTL::zz_pX &min_poly,
		   const std::vector<LinBox::DenseMatrix<GF>> &mat_seq, int number = 1);

    public:
	// length of the sequence:
	size_t getLength() const { return 2*ceil(D/(double)M); };
	// generic degree in matrix generator:
	size_t getGenDeg() const { return ceil(D/(double)M); };
	size_t getThreshold() const { return threshold; };
	
	/* CTOR                                              */
	Block_Sparse_FGLM(size_t M, InputMatrices& mat, size_t threshold);

	std::vector<NTL::zz_pX> get_lex_basis_non_generic();
	std::vector<NTL::zz_pX> get_lex_basis_generic();
	std::vector<NTL::zz_pX> get_lex_basis();
};



#endif
