def matrix_berlekamp_massey( seq, shift=None ):
	##Input:
	#   * list 'seq' of length d containing m x n matrices over a field,
	#   * shift 'shift' = tuple of m integers,
	##Default:
	#   * uniform shift [0]*m
	##Output:
	#  the unique m x m polynomial matrix P which is 
	#   * in shift-Popov form
	#   * a basis of the K[X]-module of rank m
	#     formed by polynomial vectors p in K[X]^{1 x m} such that
	#         p S  =  q  mod X^d,
	#     for some vector q such that deg(q) < deg(p),
	#     where S = seq[d-1] + seq[d-2] X + ... + seq[0] X^{d-1}
	##Note: let us relate this to the definition of recurrence relations for matrix sequences from
	#       [Definition 2.2, Kaltofen-Yuhasz, ACM Trans. Algorithms, 2013]
	# With the notation above, P is a basis of the module of p in K[X]^{1 x m} such that
	#     sum_{0 <= i <= k}  pi  seq[j+i]  = 0  for all j in {0,...,d-1-k}
	# where k = deg(p) and p = p0 + p1 X + ... + pk X^k
	# If the sequence is indeed recurrent and enough terms have been given
	# (d has been chosen large enough), this should imply that this relation
	# actually holds for all nonnegative j

	# Set dimensions, polynomial ring, and default shift
	m = seq[0].nrows()
	n = seq[0].ncols()
	pR.<X> = seq[0].base_ring()[]
	s = copy(shift) if shift != None else [0]*m
	s = s + [max(s)]*n

	# Compute minimum generating matrix as leading submatrix of approximant basis for the reversed series
	d = len(seq)
	series = Matrix.block( [[sum( [seq[d-k-1] * X^k for k in range(d)] )],[-1]] )
	pappbas = popov_iter_appbas( series, [d]*n, s )
	return (pappbas[:m,:m],pappbas[:m,m:])

def matrix_berlekamp_massey_reverse( seq, shift=None ):
	##Input:
	#   * list 'seq' of length d containing m x n matrices over a field,
	#   * shift 'shift' = tuple of m integers,
	##Default:
	#   * uniform shift [0]*m
	##Output:
	#  the unique m x m polynomial matrix P which is 
	#   * in shift-Popov form
	#   * a basis of the K[X]-module of rank m
	#     formed by polynomial vectors p in K[X]^{1 x m} such that
	#         p S  =  q  mod X^d,
	#     for some vector q such that deg(q) < deg(p),
	#     where S = seq[0] + seq[1] X + ... + seq[d-1] X^{d-1}
	##Note: TODO relation with definition of recurrence relations in Thome 2002 ??

	# Set dimensions, polynomial ring, and default shift
	m = seq[0].nrows()
	n = seq[0].ncols()
	pR.<X> = seq[0].base_ring()[]
	s = copy(shift) if shift != None else [0]*m
	s = s + [max(s)]*n

	# Compute minimum generating matrix as leading submatrix of approximant basis for the reversed series
	d = len(seq)
	series = Matrix.block( [[sum( [seq[k] * X^k for k in range(d)] )],[-1]] )
	pappbas = popov_iter_appbas( series, [d]*n, s )
	return (pappbas[:m,:m],pappbas[:m,m:])
