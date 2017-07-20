//#define TIMINGS_ON // comment out if having timings is not relevant
//#define EXTRA_VERBOSE_ON // extra detailed printed objects, like multiplication matrix and polynomial matrices... unreadable except for very small dimensions
#define VERBOSE_ON // some objects printed for testing purposes, but not the biggest ones (large constant matrices, polynomial matrices..)
//#define NAIVE_ON
#define WARNINGS_ON // comment out if having warnings for heuristic parts is irrelevant --> should probably be 'on'
#include "block-sparse-fglm.h"
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
  
  int index = 0;

  while (getline(file, line)){
    istringstream sline(line);
    vector<string> numbers{istream_iterator<string>{sline}, istream_iterator<string>{}};
    int i = stoi(numbers[0]);
    int j = stoi(numbers[1]);
    int a_int = stoi(numbers[2]);
    if (i == D){
      index++;
    }
    else{
      GF::Element a(a_int);
      mul_mats[index].refEntry(i,j) = a;
    }
  }
  file.close();
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
#ifdef EXTRA_VERBOSE_ON
	cout << "Entries:" << endl;
	for (auto &i: mat_seq)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
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
	PolMatDom::MatrixP lfac(PMD.field(),M,M,M*this->getLength());
	PolMatDom::MatrixP rfac(PMD.field(),M,M,M*this->getLength());
	PMD.SmithForm( smith, lfac, rfac, mat_gen );
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

void PolMatDom::print_degree_matrix( const MatrixP &pmat ) const {
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

size_t PolMatDom::SmithForm( vector<PolMatDom::Polynomial> &smith, PolMatDom::MatrixP &lfac, MatrixP &rfac, const PolMatDom::MatrixP &pmat ) const {
	// Heuristic computation of the Smith form and multipliers
	// Algorithm:
	//    - compute left Hermite form hmat1 = umat pmat
	//    - compute right Hermite form hmat2 = hmat1 vmat
	//    - then return (S,U,V) = (hmat2,umat,vmat)
	// Note: this is not guaranteed to be correct, but seems to be in most cases
	// Implementation: Hermite form computed via kernel basis, itself computed via approximant basis
	const size_t M = pmat.rowdim();
	const size_t deg = pmat.degree();

	// build Hermite kernel shift:  [0,...,0,0,M deg, 2 M deg, ..., (M-1) M deg]
	vector<size_t> shift( 2*M, 0 );
	for ( size_t i=M; i<2*M; ++i ) {
		shift[i] = (i-M)*(deg*M+1);
	}

	// order d such that approximant basis contains kernel basis
	const size_t order = 2*M*deg+1;

	// build series matrix: block matrix with first M rows = pmat; last M rows = -identity
	PolMatDom::MatrixP series( this->field(), 2*M, M, order ); // requirement of LinBox: degree of series = order-1 (even though here the actual degree is deg)

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

	// compute approximant basis and extract kernel basis
	PolMatDom::MatrixP app_bas( this->field(), 2*M, 2*M, order );
	OrderBasis<GF> OB( this->field() );
	OB.PM_Basis( app_bas, series, order, shift );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(Smith)### First approximant basis: shifted row degrees and matrix degrees:" << endl;
	cout << shift << endl;
	this->print_degree_matrix(app_bas);
#ifdef EXTRA_VERBOSE_ON
	cout << "basis entries:" << endl;
	cout << app_bas << endl;
#endif
#endif

	return 0;
}

template<typename Matrix>
void PolMatDom::MatrixBerlekampMassey( PolMatDom::MatrixP &mat_gen, PolMatDom::MatrixP &mat_num, const std::vector<Matrix> & mat_seq ) const {
	// 0. initialize dimensions, shift, matrices
	size_t M = mat_seq[0].rowdim();
	size_t d = mat_seq.size();
	OrderBasis<GF> OB( this->field() );
	vector<size_t> shift( 2*M, 0 );  // dim = M + N = 2M
	PolMatDom::MatrixP series( this->field(), 2*M, M, d );
	PolMatDom::MatrixP app_bas( this->field(), 2*M, 2*M, d );

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
	OB.PM_Basis( app_bas, series, d, shift );
#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Approximant basis: output shift and basis degrees" << endl;
	cout << shift << endl;
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
		if ( shift[i] != shift[i+1] )
			test = false;
	if (not test) {
		cout << "~~~WARNING(MatrixBM)~~~ unexpected degrees in approximant basis" << endl;
		cout << "   ------->>> rest of the computation may give wrong results." << endl;
	}
#endif

	// 3. Transform app_bas into Popov form
	// Right now, assuming the row degree 'shift' is constant
	// --> suffices to left-multiply by invert of leading matrix

	// retrieve inverse of leading matrix
	Matrix lmat( this->field(), 2*M, 2*M );
	for ( size_t i=0; i<2*M; ++i )
	for ( size_t j=0; j<2*M; ++j )
		lmat.refEntry(i,j) = app_bas.get(i,j,shift[i]); // row i has degree shift[i]
#ifdef EXTRA_VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### leading matrix of reduced approximant basis:" << endl;
	cout << lmat << endl;
#endif
	this->_BMD.invin( lmat ); // lmat is now the inverse of leading_matrix(app_bas)

	// create Popov approximant basis and fill it
	PolMatDom::MatrixP popov_app_bas( this->field(), 2*M, 2*M, shift[0]+1 );  //  shift[0] = deg(app_bas)
 	for ( size_t k=0; k<d; ++k ) {
		Matrix app_bas_coeff( app_bas[k] );
		this->_BMD.mulin_right( lmat, app_bas_coeff );
		for ( size_t i=0; i<2*M; ++i )
		for ( size_t j=0; j<2*M; ++j )
			popov_app_bas.ref(i,j,k) = app_bas_coeff.getEntry(i,j);
	}

#ifdef VERBOSE_ON
	cout << "###OUTPUT(MatrixBM)### Popov approximant basis degrees:" << endl;
	this->print_degree_matrix(popov_app_bas);
#ifdef EXTRA_VERBOSE_ON
	cout << "Basis entries:" << endl;
	cout << popov_app_bas << endl;
#endif
#endif

	// 4. copy into mat_gen and mat_num
	mat_gen.setsize( shift[0]+1 );
	mat_num.setsize( shift[0]+1 );
	for ( size_t i=0; i<M; ++i )
	for ( size_t j=0; j<M; ++j )
	for ( size_t k=0; k<d; ++k )
	{
		mat_gen.ref(i,j,k) = popov_app_bas.get(i,j,k);
		mat_num.ref(i,j,k) = popov_app_bas.get(i,j+M,k);
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
	  return 0;
	}

}			
