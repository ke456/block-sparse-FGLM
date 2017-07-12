#include "block-sparse-fglm.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

Block_Sparse_FGLM::Block_Sparse_FGLM(int M, int D, GF field){
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
      int r = rand();
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

int main(){
	GF field(13);
	Block_Sparse_FGLM l(2,1000,field);
	l.find_lex_basis();
}
