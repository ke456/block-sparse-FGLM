#define TIMINGS_ON // to activate timings; note that these may be irrelevant if VERBOSE / EXTRA_VERBOSE are also activated
#define EXTRA_VERBOSE_ON // extra detailed printed objects, like multiplication matrix and polynomial matrices... unreadable except for very small dimensions
#define VERBOSE_ON // some objects printed for testing purposes, but not the biggest ones (large constant matrices, polynomial matrices..)
//#define NAIVE_ON
#define WARNINGS_ON // comment out if having warnings for heuristic parts is irrelevant --> should probably be 'on'
//#define SPARSITY_COUNT // shows the sparsity of the matrices
#define TEST_FGLM // testing / timing approximant basis algos
//#define TEST_APPROX // testing / timing approximant basis algos
//#define TEST_POL  // testing xgcd and division via pmbasis
#include "block-sparse-fglm.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
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

	create_random_matrix(V);

	vector<double> sparsity_count;
	long max_entries = D*D;

	int index = 0;
	long count = 0;
	while (getline(file, line)){
	istringstream sline(line);
	vector<string> numbers{istream_iterator<string>{sline}, istream_iterator<string>{}};
	int i = stoi(numbers[0]);
	int j = stoi(numbers[1]);
	int a_int = stoi(numbers[2]);
	if (i == D){
	  index++;
	  sparsity_count.emplace_back((double)(count)/max_entries);
	  count = 0;
	}
	else{
	  count++;
	  GF::Element a(a_int);
	  mul_mats[index].refEntry(i,j) = a;
	}
	}
	file.close();
	  
#ifdef SPARSITY_COUNT
	cout << "sparsity: ";
	for (auto i: sparsity_count)
		cout << i << " ";
	cout << endl;
#endif
#ifdef EXTRA_VERBOSE_ON
	for (auto &i : mul_mats)
		i.write(cout << "mul mat", Tag::FileFormat::Maple)<<endl;
#endif
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
 int c,
 size_t to){
	// gather all the matrices of l in a single (seq_length*M) x D matrix
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
	DenseMatrix<GF> result(field, to*M,c);
	MD.mul(result,mat,V);
	
	v.resize(to, DenseMatrix<GF>(field,M,c));
	for (size_t i = 0; i < to; i++){
		v[i] = DenseMatrix<GF>(field, M, c);
		for (int row = 0; row < M; row++){
			int r = i * M + row;
			for (int col = 0; col < c; col++){
				GF::Element a;
				result.getEntry(a,r,col);
				v[i].refEntry(row,col) = a;
			}
		}
	}
}

// reverses every entry of mat at degree d
//PolMatDom::PMatrix matrix_poly_reverse
//( const PolMatDom::PMatrix &mat, const int d){
//}

//shifts every entry by d
void shift(PolMatDom::PMatrix &result, const PolMatDom::PMatrix &mat, 
int row, int col, int deg){
	for (int i = 0; i < row; i++)
	  for (int j = 0; j < col; j++)
		  for (int d = 0; i < deg; i++){
				auto element = mat.get(i,j,d+deg);
				result.ref(i,j,d) = element;
			}
}

void Block_Sparse_FGLM::find_lex_basis(){
#ifdef TIMINGS_ON
	Timer tm;
	tm.clear(); tm.start();
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
	tm.stop();
	cout << "###TIME### left sequence (UT1^i): " << ": " << tm.usertime() << " (user time)" << endl;
	cout << "###TIME### left sequence (UT1^i): " << tm.realtime() << " (real time)" << endl;
	tm.clear(); tm.start();
#endif
	vector<DenseMatrix<GF>> mat_seq(getLength(), DenseMatrix<GF>(field,M,M));
	// 2. compute the total matrix sequence (UT1^i)V
	get_matrix_sequence(mat_seq, mat_seq_left, V, M, getLength());
#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### sequence (UT1^i)V: " << ": " << tm.usertime() << endl;
	tm.clear(); tm.start();
#endif
#ifdef VERBOSE_ON
  cout << "###OUTPUT### Matrix V:" << endl;
	V.write(cout,Tag::FileFormat::Maple)<<endl;
	cout << "###OUTPUT### Matrix sequence (U T1^i V)_i :" << endl;
	cout << "Length d = " << this->getLength() << endl;
	cout << "Generic generator degree: deg = " << this->getGenDeg() << endl;
#ifdef EXTRA_VERBOSE_ON
	cout << "Entries:" << endl;
	for (auto &i: mat_seq)
		i.write(cout, Tag::FileFormat::Maple)<<endl;
#endif
#endif
	// 3. compute generator and numerator in Matrix Berlekamp Massey: reversed sequence = mat_gen/mat_num
	// note: * generator is in Popov form
	//       * degree of row i of numerator < degree of row i of generator
	PolMatDom PMD( field );
	PolMatDom::MatrixP mat_gen(PMD.field(),M,M,this->getLength());
	PolMatDom::MatrixP mat_num(PMD.field(),M,M,this->getLength());
	PMD.MatrixBerlekampMassey<DenseMatrix<GF>>( mat_gen, mat_num, mat_seq );
#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Matrix Berlekamp Massey: " << tm.usertime() << endl; 
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
#ifdef TIMINGS_ON
	tm.clear(); tm.start();
#endif
	vector<PolMatDom::Polynomial> smith( M );
	PolMatDom::MatrixP lfac(PMD.field(),M,M,M*this->getLength()+1);
	PolMatDom::MatrixP rfac(PMD.field(),M,M,M*this->getLength()+1);
	PMD.SmithForm( smith, lfac, rfac, mat_gen );
#ifdef TIMINGS_ON
	tm.stop();
	cout << "###TIME### Smith form and transformations: " << tm.usertime() << endl; 
#endif

// finding u_tilde

PolMatDom::MatrixP w(PMD.field(),1,M,M*this->getLength()+1);
for (int i = 0; i < M; i++)
for (int j = 0; j < M*this->getLength()+1; j++){
	auto element = rfac.get(M-1,i,j);
	w.ref(0,i,j) = element;
}
cout << "W: " << w << endl;

MatrixDomain<GF> MD(field);
#ifdef NAIVE_ON
	tm.clear(); tm.start();
	DenseMatrix<GF> U(field,M,D);
	
	vector<DenseMatrix<GF>> lst(this->getLength(), DenseMatrix<GF>(field,M,D));
	auto &T1 = mul_mats[0];
	create_random_matrix(U);
	lst[0] = U;
	for ( size_t i=1; i<this->getLength(); i++){
		MD.mul(lst[i],lst[i-1],T1);
	}
#ifdef EXTRA_VERBOSE_ON
	for (auto &i : lst)
		i.write(cout<<"U: ", Tag::FileFormat::Maple)<<endl;
#endif
	tm.stop();
	cout << "###TIME### sequence (UT1^i) naive: " << tm.usertime() << endl;
#endif
	DenseMatrix<GF> V_col(field,D,1); // a single column of V
	for (int i = 0; i < D; i++){
		auto el = V.refEntry(i,0);
		V_col.refEntry(i,0) = el;
	}
	V_col.write(cout<<"V_col:"<<endl,Tag::FileFormat::Maple)<<endl;
  PolynomialMatrixMulDomain<GF> PMMD(field);
  // LOOP FOR OTHER VARIABLES
  for (int i  = 1; i < mul_mats.size(); i++){
		DenseMatrix<GF> right_mat(field, D, 1); // Ti * V (only one column)
		auto &Ti = mul_mats[i];
		MD.mul(right_mat, Ti, V_col);
		vector<DenseMatrix<GF>> seq(this->getGenDeg(),DenseMatrix<GF>(field,M,1));
		get_matrix_sequence(seq,mat_seq_left,right_mat,1,getGenDeg());
		for (auto &i : seq)
		  i.write(cout << "seq:"<<endl, Tag::FileFormat::Maple)<<endl;
		PolMatDom::PMatrix polys(PMD.field(),M,1,this->getGenDeg());
		// creating the poly matrix from the sequence in reverse order
		int index = 0;
		for (int j = seq.size()-1; j >= 0; j--){
			for (int q = 0; q < M; q++){
				auto element = seq[j].refEntry(q,0);
				polys.ref(q,0,index)= element;
			}
			index++;
		}
		cout << "Poly: " << endl << polys << endl;
		PolMatDom::PMatrix N(PMD.field(),M,1,this->getLength());
		PMMD.mul(N,mat_gen,polys);
		cout << "N:" << endl << N << endl;
		shift(N,N,M,1,getGenDeg());
		cout << "Shifted N: " << endl<< N << endl;
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

	cout << "s=" << s<< endl;
	if (s == ""){
	  GF field(p);
#ifdef TEST_FGLM
	  Block_Sparse_FGLM l(field, D, M, n);
	  l.find_lex_basis();
#endif
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
		cout << "blocking dimension:" << " M=" << M << endl;
		file.close();
		GF field(p);
#ifdef TEST_FGLM
		Block_Sparse_FGLM l(field, D, M, n, s);
		l.find_lex_basis();
#endif
	}
	GF field(p);
	long seed = time(NULL);
	typename GF::RandIter rd(field,0,seed);
	PolMatDom PMD( field );
#ifdef TEST_APPROX
	cout << "~~~~~~~~~~~STARTING TESTS APPROXIMANTS~~~~~~~~~~~~~" << endl;
	size_t order = 2*ceil(D/(double)M);
	cout << "order for approximant basis : " << order << endl;
	cout << "dimensions of input series matrix : " << 2*M << " x " << M << endl;
	PolMatDom::PMatrix series( field, 2*M, M, order );
	for ( size_t d=0; d<order; ++d )
	for ( size_t i=0; i<2*M; ++i )
	for ( size_t j=0; j<M; ++j )
		rd.random( series.ref( i, j, d ) );
	const vector<int> shift( 2*M, 0 );
	Timer tm;
  {
    cout << "~~~NOW TESTING OLD MBASIS~~~" << endl;
    PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
    tm.clear(); tm.start();
    vector<int> rdeg1 = PMD.old_mbasis( app_bas, series, order, shift );
    tm.stop();
#ifdef VERBOSE_ON
    cout << "###OUTPUT### degrees in approx basis:" << endl;
    PMD.print_degree_matrix( app_bas );
#endif
    cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
    cout << "###TIME### appprox basis: " << tm.usertime() << endl;
  }
  {
    cout << "~~~NOW TESTING OLD MBASIS~~~" << endl;
    PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
    tm.clear(); tm.start();
    vector<int> rdeg2 = PMD.old_pmbasis( app_bas, series, order, shift );
    tm.stop();
#ifdef VERBOSE_ON
    cout << "###OUTPUT### degrees in approx basis:" << endl;
    PMD.print_degree_matrix( app_bas );
#endif
    cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
    cout << "###TIME### appprox basis: " << tm.usertime() << endl;
  }
  {
    cout << "~~~NOW TESTING NEW MBASIS~~~" << endl;
    PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
    tm.clear(); tm.start();
    vector<size_t> mindeg3 = PMD.mbasis( app_bas, series, order, shift );
    tm.stop();
#ifdef VERBOSE_ON
    cout << "###OUTPUT### degrees in approx basis:" << endl;
    PMD.print_degree_matrix( app_bas );
#endif
    cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
    cout << "###TIME### appprox basis: " << tm.usertime() << endl;
  }
  {
    cout << "~~~NOW TESTING NEW PMBASIS~~~" << endl;
    PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
    tm.clear(); tm.start();
    vector<size_t> mindeg4 = PMD.pmbasis( app_bas, series, order, shift );
    tm.stop();
#ifdef VERBOSE_ON
    cout << "###OUTPUT### degrees in approx basis:" << endl;
    PMD.print_degree_matrix( app_bas );
#endif
    cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
    cout << "###TIME### appprox basis: " << tm.usertime() << endl;
  }
  {
    cout << "~~~NOW TESTING POPOV_PMBASIS~~~" << endl;
    PolMatDom::PMatrix app_bas( field, 2*M, 2*M, order );
    tm.clear(); tm.start();
    vector<size_t> mindeg5 = PMD.popov_pmbasis( app_bas, series, order, shift );
    tm.stop();
#ifdef VERBOSE_ON
    cout << "###OUTPUT### degrees in approx basis: " << endl;
    PMD.print_degree_matrix( app_bas );
#endif
    cout << "###CORRECTNESS### has required order: " << test_order( app_bas, series, order ) << endl;
    cout << "###TIME### appprox basis: " << tm.usertime() << endl;
  }
#endif
#ifdef TEST_POL
  Timer tm2;
	cout << "~~~~~~~~~~~STARTING TESTS POLY~~~~~~~~~~~~~" << endl;
  {
    cout << "small xgcd with coprime polynomials" << endl;
    PolMatDom::Polynomial a = {1,1,-1,0,-1,1};
    PolMatDom::Polynomial b = {-1,0,1,1,0,0,-1};
    PolMatDom::Polynomial g,u,v;
    // a = 1+X-X^2-X^4+X^5
    // b = -1 + X^2 + X^3 -X^6
    // xgcd(a,b) = (1,
    //              15379115*X^5 + 15379116*X^2 + 7689558*X + 7689558,
    //              15379115*X^4 + 7689558*X^3 + 15379116*X + 7689557)
    // when over Z/pZ with p = 23068673 
    tm2.clear(); tm2.start();
    PMD.xgcd(a,b,g,u,v);
    tm2.stop();
    cout << "###TIME### xgcd: " << tm2.usertime() << endl;
#ifdef VERBOSE_ON
    cout << "###OUTPUT### xgcd input: " << endl;
    cout << a << endl;
    cout << b << endl;
    cout << "###OUTPUT### xgcd output: " << endl;
    cout << g << endl;
    cout << u << endl;
    cout << v << endl;
#endif
  }
  {
    cout << "small xgcd with gcd of degree 1" << endl;
    PolMatDom::Polynomial a = {59,62,23068617,23068670,23068614,56,3};
    PolMatDom::Polynomial b = {23068614,23068670,59,62,3,0,23068614,23068670};
    PolMatDom::Polynomial g,u,v;
    // same polynomials multiplied by 3*X + 59
    // a = 3*X^6 + 56*X^5 + 23068614*X^4 + 23068670*X^3 + 23068617*X^2 + 62*X + 59
    // b = 23068670*X^7 + 23068614*X^6 + 3*X^4 + 62*X^3 + 59*X^2 + 23068670*X + 23068614
    // xgcd(a,b) = (X + 15379135,
    //              20505487*X^5 + 5126372*X^2 + 2563186*X + 2563186,
    //              20505487*X^4 + 2563186*X^3 + 5126372*X + 17942301)
    // when over Z/pZ with p = 23068673 
    tm2.clear(); tm2.start();
    PMD.xgcd(a,b,g,u,v);
    tm2.stop();
    cout << "###TIME### xgcd: " << tm2.usertime() << endl;
#ifdef VERBOSE_ON
    cout << "###OUTPUT### xgcd input: " << endl;
    cout << a << endl;
    cout << b << endl;
    cout << "###OUTPUT### xgcd output: " << endl;
    cout << g << endl;
    cout << u << endl;
    cout << v << endl;
#endif
  }
  {
    cout << "xgcd with random polynomials of degree: " << D-1 << endl;
    PolMatDom::Polynomial a( D );
    PolMatDom::Polynomial b( D );
    for ( size_t d=0; d<D; ++d )
    {
      rd.random( a[d] );
      rd.random( b[d] );
    }
    PolMatDom::Polynomial g,u,v;
    // same polynomials multiplied by 3*X + 59
    // a = 3*X^6 + 56*X^5 + 23068614*X^4 + 23068670*X^3 + 23068617*X^2 + 62*X + 59
    // b = 23068670*X^7 + 23068614*X^6 + 3*X^4 + 62*X^3 + 59*X^2 + 23068670*X + 23068614
    // xgcd(a,b) = (X + 15379135,
    //              20505487*X^5 + 5126372*X^2 + 2563186*X + 2563186,
    //              20505487*X^4 + 2563186*X^3 + 5126372*X + 17942301)
    // when over Z/pZ with p = 23068673 
    tm2.clear(); tm2.start();
    PMD.xgcd(a,b,g,u,v);
    tm2.stop();
    cout << "###TIME### xgcd: " << tm2.usertime() << endl;
  }
  {
    cout << "division with quotient of degree 1" << endl;
    PolMatDom::Polynomial a = {59,62,23068617,23068670,23068614,56,3};
    PolMatDom::Polynomial b = {1,1,-1,0,-1,1};
    PolMatDom::Polynomial q;
    // a = 3*X^6 + 56*X^5 + 23068614*X^4 + 23068670*X^3 + 23068617*X^2 + 62*X + 59
    // b = 1+X-X^2-X^4+X^5
    // quotient should be X + 15379135,
    // when over Z/pZ with p = 23068673 
    tm2.clear(); tm2.start();
    PMD.divide(a,b,q);
    tm2.stop();
    cout << "###TIME### divide: " << tm2.usertime() << endl;
#ifdef VERBOSE_ON
    cout << "###OUTPUT### divide input: " << endl;
    cout << a << endl;
    cout << b << endl;
    cout << "###OUTPUT### divide output: " << endl;
    cout << q << endl;
#endif
  }
#endif
	return 0;
}
