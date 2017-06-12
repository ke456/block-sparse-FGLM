def iter_appbas( serie, order, shift=None ):
	##Input:
	#   * m x n matrix of polynomials 'serie',
	#   * an integer 'order',
	#   * degree shift 'shifts' (list of m integers)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output: an approximant basis 'appbas' for 'serie' at order X^order
	# in shift-ordered weak Popov form
	##References for such iterative approximant basis algorithms:
	# [van Barel-Bultheel, 1992, Numer. Algor.]
	# [Beckermann-Labahn, 1994, SIAM J. Matrix Anal. & Appl.]

	#polynomial ring, initialization of the residuals and the approximants
	polRing,(X,) = serie.base_ring().objgens()
	appbas = Matrix.identity( polRing, serie.nrows() )
	residuals = copy(serie)

	# set up the shift 'rdeg' that will be updated through the algorithm
	# --> 'rdeg' is initially the shift-row degree of the identity matrix
	# recall default = uniform shift --> rdeg = 0-row degree of Id = [0,...,0]
	rdeg = copy(shift) if shift != None else [0]*serie.nrows()

	for sigma in range(order): 
		# invariant: appbas * serie = residuals
		#with residuals == 0 mod X^sigma
		#and appbas is shift-ordered weak Popov with shift-row degree rdeg

		#coefficients of degree sigma of the residuals
		constRes = Matrix( [ [residuals[i,j][sigma] 
			for j in range( residuals.ncols() )]
			for i in range( residuals.nrows() )] )

		for j in range( serie.ncols() ):
			# invariant: appbas * serie = residuals 
			#with residuals[:,0:j] == 0 mod X^(sigma+1) 
			#and residuals[:,j:n] == 0 mod X^sigma
			#and appbas is shift-ordered weak Popov with shift-row degree rdeg

			#Lambda: collect rows with nonzero coefficient
			#pi: index of the first row with smallest shift (among those in Lambda)
			Lambda = []
			pi = -1
			for i in range( constRes.nrows() ):
				if constRes[i,j] != 0:
					Lambda.append(i)
					if pi < 0 or rdeg[i] < rdeg[pi]:
						pi = i
			if Lambda != [] : # otherwise nothing to do
				#update all rows in Lambda--{pi}
				Lambda.remove(pi)
				for row in Lambda:
					scalar = -constRes[row,j]/constRes[pi,j]
					appbas.add_multiple_of_row( row, pi, scalar )
					residuals.add_multiple_of_row( row, pi, scalar )
					constRes.add_multiple_of_row( row, pi, scalar )
				#update row pi
				rdeg[pi] += 1
				appbas.rescale_row( pi, X )
				residuals.rescale_row( pi, X )
				constRes.rescale_row( pi, 0 )
	return appbas

def leading_matrix_row( mat, shift=None ):
	# input: m x n polynomial matrix 'mat', degree shift
	# output: shift-leading matrix of mat (working in rows)
	# assumption: pmat has no zero row
	m = mat.nrows()
	n = mat.ncols()
	lmat = Matrix( mat.base_ring().base_ring(), m, n )
	if shift==None:
		shift = [0]*n
	for i in range(m):
		# compute shifted row degree
		rdeg = max( [mat[i,j].degree() + shift[j] for j in range(n) if mat[i,j] != 0 ] )
		# row i of lmat = entries of row i of mat achieving this rdeg
		for j in range(n):
			if ( mat[i,j] != 0 and mat[i,j].degree() + shift[j] == rdeg ):
				lmat[i,j] = mat[i,j].leading_coefficient()
	return lmat

def iter_popov_appbas( serie, order, shift=None ):
	##Input:
	#   * m x n matrix of polynomials 'serie',
	#   * an integer 'order',
	#   * degree shift 'shifts' (list of m integers)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output: the unique approximant basis 'appbas' for 'serie'
	#at order X^order which is in shift-Popov form
	appbas = iter_appbas( serie, order, shift )
	mindeg = [appbas[i,i].degree() for i in range(appbas.nrows())]
	reduced = iter_appbas( serie, order, [-mindeg[i] for i in range(len(mindeg))] )
	return leading_matrix_row( reduced, [-mindeg[i] for i in range(len(mindeg))] ).I * reduced


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
	pR.<X> = seq[0].base_ring()[]
	s = copy(shift) if shift != None else [0]*seq[0].nrows()
	s = s + [max(s)]*seq[0].ncols()

	# Compute minimum generating matrix as leading submatrix of approximant basis for the reversed series
	d = len(seq)
	series = Matrix.block( [[sum( [seq[d-k-1] * X^k for k in range(d)] )],[-1]] )
	pappbas = iter_popov_appbas( series, d, s )
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
	pR.<X> = seq[0].base_ring()[]
	s = copy(shift) if shift != None else [0]*seq[0].nrows()
	s = s + [max(s)]*seq[0].ncols()

	# Compute minimum generating matrix as leading submatrix of approximant basis for the reversed series
	d = len(seq)
	series = Matrix.block( [[sum( [seq[k] * X^k for k in range(d)] )],[-1]] )
	pappbas = iter_popov_appbas( series, d, s )
	return (pappbas[:m,:m],pappbas[:m,m:])
