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

void Block_Sparse_FGLM::find_lex_basis(){
	MatrixDomain<Givaro::Modular<int>> MD{field};
	vector<DenseMatrix<GF>> U_rows(M, DenseMatrix<GF>(field,1,D));
	for (auto &i : U_rows)
		create_random_matrix(i);

	vector<vector<DenseMatrix<GF>>> mat_seq(D);
	for (auto &i : mat_seq){
		i = vector<DenseMatrix<GF>>(M, DenseMatrix<GF>(field,1,D));
	}
	
  DenseMatrix<GF> T1(field,D,D);
  create_random_matrix(T1);
  //T1.write(cout, Tag::FileFormat::Maple) << endl;
  
  //for (auto &i: U_rows)
  //	i.write(cout, Tag::FileFormat::Maple) << endl;
  
  auto start = clock();
  
  omp_set_num_threads(M);
  #pragma omp parallel for
	for (int i  = 0; i < M; i++){
		mat_seq[0][i] = U_rows[i];
		for (int j  = 1; j < ceil(D/(M*1.0)); j++){
			//cout << "i,j: " << i << " " << j << endl;
			auto &l = mat_seq[j-1][i];
			auto &result = mat_seq[j][i];
			MD.mul(result,l,T1);
		}
	}
	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Parallel took: " << duration << endl;
	
	start = clock();
	for (int i  = 0; i < 1; i++){
		mat_seq[0][i] = U_rows[i];
		for (int j  = 1; j < D; j++){
			//cout << "i,j: " << i << " " << j << endl;
			auto &l = mat_seq[j-1][i];
			auto &result = mat_seq[j][i];
			MD.mul(result,l,T1);
		}
	}
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Direct took: " << duration << endl;
	
	//for (auto &i : mat_seq){
	//	cout << "New power:" << endl;
	//	for (auto &j : i)
	//		j.write(cout, Tag::FileFormat::Maple)<< endl;
	//}
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
