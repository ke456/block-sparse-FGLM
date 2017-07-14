#define TIMINGS_ON // comment out if having timings is not relevant
//#define VERBOSE_ON // comment out unless you want many object printed (e.g. for testing purposes)
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

Block_Sparse_FGLM::Block_Sparse_FGLM(int M, int D, const GF &field): V(field,D,M){
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
	MatrixDomain<GF> MD{field};

	// initialize the left block (random MxD)
	vector<DenseMatrix<GF>> U_rows(M, DenseMatrix<GF>(field,1,D));
	for (auto &i : U_rows){
		create_random_matrix(i);
#ifdef VERBOSE_ON
		i.write(cout << "###OUTPUT### U_row: ",Tag::FileFormat::Maple)<<endl;
#endif
	}
	
	// stores U_i*T1^j at mat_seq[i][j]
	vector<vector<DenseMatrix<GF>>> mat_seq(M);
	for (auto &i : mat_seq){
		i = vector<DenseMatrix<GF>>(this->getLength(),DenseMatrix<GF>(field,1,D));
	}

	// initialize the first multiplication matrix (random DxD)
	DenseMatrix<GF> T1(field,D,D);
	create_random_matrix(T1);
#ifdef VERBOSE_ON
	T1.write(cout << "###OUTPUT### Multiplication matrix T1:"<<endl, Tag::FileFormat::Maple)<<endl;
#endif

	// 1st version: compute sequence in a parallel fashion
#pragma omp parallel for num_threads(M)
	for (int i  = 0; i < M; i++){
		MatrixDomain<GF> MD2{field};
		vector<DenseMatrix<GF>> temp_mat_seq(this->getLength(), DenseMatrix<GF>(field,1,D)); 
		temp_mat_seq[0] = U_rows[i];
		auto &T1_temp = T1;
		for (size_t j  = 1; j < this->getLength(); j++){
			auto &l = temp_mat_seq[j-1];
			auto &result = temp_mat_seq[j];
			MD2.mul(result,l,T1_temp);
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

void Block_Sparse_FGLM::get_matrix_sequence(vector<DenseMatrix<GF>> &v){
	// gather all the matrices of v in a single (seq_length*M) x D matrix
	MatrixDomain<GF> MD(field);
	DenseMatrix<GF> mat(field, this->getLength()*M, D);
	for (size_t i = 0; i < this->getLength(); i++){
		auto &m = v[i];
		for (int row = 0; row < M; row++){
			int r = i * M + row; // starting point for mat
			for (int col = 0; col < D; col++){
				GF::Element a;
				m.getEntry(a,row,col);
				mat.refEntry(r,col) = a;
			}
		}
	}
	create_random_matrix(V);
	DenseMatrix<GF> result(field, this->getLength()*M,M);
	MD.mul(result,mat,V);
	
	for (size_t i = 0; i < this->getLength(); i++){
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
	vector<DenseMatrix<GF>> mat_seq(this->getLength(), DenseMatrix<GF>(field,M,D));
	get_matrix_sequence_left(mat_seq);
#ifdef TIMINGS_ON
	auto end = chrono::high_resolution_clock::now();
	cout << "###TIME### left sequence (UT1^i): " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
	start = chrono::high_resolution_clock::now();
#endif
	// 2. compute the total matrix sequence (UT1^i)V
	get_matrix_sequence(mat_seq);
#ifdef TIMINGS_ON
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### sequence (UT1^i)V: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
	start = chrono::high_resolution_clock::now();
#endif
#ifdef VERBOSE_ON
	cout << "###OUTPUT### Matrix sequence (U T1^i V)_i :" << endl;
	cout << "length d = " << this->getLength() << ", and entries:" << endl;
	cout << mat_seq << endl;
#endif
	// 3. compute generator and numerator in Matrix Berlekamp Massey: reversed sequence = mat_gen/mat_num
	PolMatDom PMD( field );
	PolMatDom::MatrixP mat_gen(PMD.field(),M,M,this->getLength());
	PolMatDom::MatrixP mat_num(PMD.field(),M,M,this->getLength());
	PMD.MatrixBerlekampMassey<DenseMatrix<GF>>( mat_gen, mat_num, mat_seq );
#ifdef TIMINGS_ON
	end = chrono::high_resolution_clock::now();
	cout << "###TIME### Matrix Berlekamp Massey: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl; 
#endif
#ifdef VERBOSE_ON
  	cout << "###OUTPUT### Matrix generator:" << endl;
	cout << mat_gen << endl;
  	cout << "###OUTPUT### Matrix numerator:" << endl;
	cout << mat_num << endl;
#endif
}

int main( int argc, char **argv ){
	// default arguments
	//size_t p = 13;  // size of the base field
	size_t p = 23068673;  // size of the base field
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


void PolMatDom::print_pmat( const PolMatDom::MatrixP &pmat ) const {
	for ( size_t i=0; i<pmat.rowdim(); ++i )
	{
		for ( size_t j=0; j<pmat.coldim(); ++j )
		{
			for ( size_t d=0; d<pmat.size()-1; ++d )
				cout << pmat.get(i,j,d) << "X^" << d << '+';
			cout << pmat.get(i,j,pmat.size()-1) << "X^" << pmat.size()-1 << '\t';
		}
		cout << endl;
	}
}

size_t PolMatDom::SmithForm( vector<PolMatDom::Polynomial> &smith, PolMatDom::MatrixP &lfac, MatrixP &rfac, const PolMatDom::MatrixP &pmat ) const {
	return 0;
}

template<typename Matrix>
void PolMatDom::MatrixBerlekampMassey( PolMatDom::MatrixP &mat_gen, PolMatDom::MatrixP &mat_num, const std::vector<Matrix> & mat_seq ) const {
	// initialize dimensions, shift, matrices
	size_t M = mat_seq[0].rowdim();
	size_t d = mat_seq.size();
	OrderBasis<GF> OB( this->field() );
	vector<size_t> shift( 2*M, 0 );  // dim = M + N = 2M
	PolMatDom::MatrixP series( this->field(), 2*M, M, d );
	PolMatDom::MatrixP app_bas( this->field(), 2*M, 2*M, d );

	// construct series = Matrix.block( [[sum( [seq[d-k-1] * X^k for k in range(d)] )],[-1]] )
	// i.e. stacking reversed sequence and -Identity
	for ( size_t i=0; i<M; ++i )
		series.ref(i+M,i,0) = this->field().mOne;

	for ( size_t k=0; k<d; ++k )
	for ( size_t i=0; i<M; ++i )
	for ( size_t j=0; j<M; ++j )
		series.ref(i,j,k) = mat_seq[d-k-1].getEntry(i,j);

#ifdef VERBOSE_ON
	cout << "###OUTPUT### Series input to approximant basis:" << endl;
	cout << series << endl;
#endif

	// compute approximant and copy into mat_gen,mat_num
	OB.PM_Basis( app_bas, series, d, shift );
#ifdef VERBOSE_ON
	cout << "###OUTPUT### Computed approximant basis:" << endl;
	cout << series << endl;
#endif

	for ( size_t i=0; i<M; ++i )
	for ( size_t j=0; j<M; ++j )
	for ( size_t k=0; k<d; ++k )
	{
		mat_gen.ref(i,j,k) = app_bas.get(i,j,k);
		mat_num.ref(i,j,k) = app_bas.get(i,j+M,k);
	}
}
