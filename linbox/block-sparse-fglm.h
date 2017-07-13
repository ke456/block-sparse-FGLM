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

	/* Helpers                                           */
	template<typename Matrix>
	void create_random_matrix(Matrix &m);
	
	// Computes sequence (UT1^i)
	void get_matrix_sequence_left(std::vector<LinBox::DenseMatrix<GF>> &);
	
	// Given a sequence holding U_i*T1^j as the (i,j)-th entry, produces
	// sequence of matrices such that the i-th entry is U*T1^i
	void gather(std::vector<LinBox::DenseMatrix<GF>> &v);

	public:
	/* CTOR                                              */
	Block_Sparse_FGLM(int,int,const GF &);

	void find_lex_basis();
};

#endif
