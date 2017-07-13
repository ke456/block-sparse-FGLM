#include "block-sparse-fglm.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace LinBox;
using namespace std;

Block_Sparse_FGLM::Block_Sparse_FGLM(int M, int D, const GF &field){
	this->M = M;
	this->D = D;
	this->field = field;
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
	MatrixDomain<Givaro::Modular<int>> MD{field};

	// initialize the left block (random MxD)
	vector<DenseMatrix<GF>> U_rows(M, DenseMatrix<GF>(field,1,D));
	for (auto &i : U_rows)
		create_random_matrix(i);
	
	// stores U_i*T1^j at mat_seq[i][j]
	vector<vector<DenseMatrix<GF>>> mat_seq(M);
	for (auto &i : mat_seq){
		i = vector<DenseMatrix<GF>>(ceil(D/(double)M),DenseMatrix<GF>(field,1,D));
	}

	// initialize the first multiplication matrix (random DxD)
	DenseMatrix<GF> T1(field,D,D);
	create_random_matrix(T1);

	auto start = clock();

	// 1st version: compute sequence in a parallel fashion
#pragma omp parallel for num_threads(M)
	for (int i  = 0; i < M; i++){
		MatrixDomain<Givaro::Modular<int>> MD2{field};
		vector<DenseMatrix<GF>> temp_mat_seq(ceil(D/(double)M), DenseMatrix<GF>(field,1,D)); 
		temp_mat_seq[0] = U_rows[i];
		auto &T1_temp = T1;
		for (int j  = 1; j < ceil(D/(M*1.0)); j++){
			auto &l = temp_mat_seq[j-1];
			auto &result = temp_mat_seq[j];
			MD2.mul(result,l,T1_temp);
		}
		mat_seq[i] = temp_mat_seq;
	}
	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Parallel took: " << duration << endl;
}

void Block_Sparse_FGLM::find_lex_basis(){
}

int main( int argc, char **argv ){
	// default arguments
	size_t p = 13;  // size of the base field
	size_t M = 4;   // row dimension for the blocks
	//size_t N = 4;   // column dimension for the blocks (useless right now: fixed to M)
	size_t D = 512; // vector space dimension / dimension of multiplication matrices

	static Argument args[] = {
		{ 'p', "-p p", "Set cardinality of the base field to p.", TYPE_INT, &p },
		{ 'M', "-M M", "Set the row block dimension to M.", TYPE_INT,       &M },
		//{ 'N', "-N N", "Set the column block dimension to N.", TYPE_INT,       &N },
		{ 'D', "-D D", "Set dimension of test matrices to MxN.", TYPE_INT,  &D },
		END_OF_ARGUMENTS
	};	

	parseArguments (argc, argv, args);

	GF field(p);
	Block_Sparse_FGLM l(M,D,field);
	l.find_lex_basis();
}
