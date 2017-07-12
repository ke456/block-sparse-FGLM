#include <linbox/matrix/sparse-matrix.h>
#include <linbox/matrix/dense-matrix.h>
#include <linbox/matrix/matrix-domain.h>
#include <linbox/integer.h>
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;
using namespace LinBox;

typedef Givaro::Modular<int> GF;
class Block_Sparse_FGLM{
	// stores the multiplication matrices T_i
	vector<SparseMatrix<GF>> mul_mats;
	// the current field
	GF field;
	
	int D; // the degree of the minpoly (also # of mul-mats)
	int M; // number of blocks (set to number of CPUs?)
	
	/* Helpers                                           */
	template<typename Matrix>
	void create_random_matrix(Matrix &m);
		
	public:
	/* CTOR                                              */
	Block_Sparse_FGLM(int,int,GF);

	void find_lex_basis();
};
