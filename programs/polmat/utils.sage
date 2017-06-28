#### Some useful functions:
##   - matrix-wise polynomial operations (degree, reverse, shift, truncate..)
##   - degrees (max degree, column/row degree, and shifted variants)
##   - creating random matrices (with given degree, in Popov form, in weak Popov form..)
##   - shifted row-reducedness (leading matrix, tests of reducedness, of shifted weak Popov..)
##   - constant matrix decompositions (LSP, LQUP, REF, ..)

### MATRIX-WISE POLYNOMIAL OPERATIONS ###
def degree_matrix( A ):
	#input: polynomial matrix 'A' and degree shift 'shift'
	#output: integer matrix of same dimension of 'A' with [i,j] entry the degree of A[i,j]
	return Matrix( ZZ, [[A[i,j].degree() for j in range(A.ncols())] for i in range(A.nrows())] )

def valuation_matrix( A ):
	#input: polynomial matrix 'A' and degree shift 'shift'
	#output: integer matrix of same dimension of 'A' with [i,j] entry the degree of A[i,j]
	return Matrix( ZZ, [[A[i,j].valuation() if A[i,j].valuation() != +Infinity else -1 for j in range(A.ncols()) ] for i in range(A.nrows())] )

def matrix_reverse( A, d=None ):
	# input: polynomial matrix A(X) and nonnegative integer d bounding the degree of A
	# output: X^d A(X^-1) 
	m = A.nrows()
	n = A.ncols()
	Arev = Matrix( A.base_ring(), m, n )
	if d == None:
		d = matrix_degree(A)
	for j in range(n):
		for i in range(m):
			Arev[i,j] = A[i,j].reverse(d)
	return Arev

def matrix_reverse_rows( A, d=None ):
	# input: polynomial matrix A(X) and tuple d bounding the row degree componentwise
	# output: diag(X^d1,...,X^dn) A(X^-1) 
	m = A.nrows()
	n = A.ncols()
	Arev = Matrix( A.base_ring(), m, n )
	if d == None:
		d = row_degree(A)
	for j in range(n):
		for i in range(m):
			Arev[i,j] = A[i,j].reverse(d[i])
	return Arev

def matrix_reverse_columns( A, d=None ):
	# input: polynomial matrix A(X) and tuple d bounding the column degree componentwise
	# output: A(X^-1) diag(X^d1,...,X^dn)
	m = A.nrows()
	n = A.ncols()
	Arev = Matrix( A.base_ring(), m, n )
	if d == None:
		d = column_degrees(A)
	for j in range(n):
		for i in range(m):
			Arev[i,j] = A[i,j].reverse(d[j])
	return Arev

def matrix_truncate( A, N ):
	# input: polynomial matrix A(X) and degree bound N
	# output: A mod X^N
	return A.apply_map( lambda p: p.truncate(N) )

def matrix_shift( A, N ):
	# input: polynomial matrix A(X) and degree shift N
	# output: A * X^N if N>=0, A//X^-N if N<0
	return A.apply_map( lambda p: p.shift(N) )

def matrix_shift_columns( A, shift ):
	# input: polynomial matrix A(X) and degree shift s
	# output: A with columns shifted by s
	B = copy(A)
	for j in range(A.ncols()):
		B[:,j] = B[:,j].apply_map( lambda p: p.shift(shift[j]) )
	return B

def matrix_shift_rows( A, shift ):
	# input: polynomial matrix A(X) and degree shift s
	# output: A with rows shifted by s
	B = copy(A)
	for i in range(A.nrows()):
		B[i,:] = B[i,:].apply_map( lambda p: p.shift(shift[j]) )
	return B

### COLUMN DEGREES AND ROW DEGREES ###
def matrix_degree( A, s=None ):
	# input: polynomial matrix A = [a_ij], degree shift s
	# assumption: A nonzero
	# output: maximum shifted degree  max_ij ( deg(a_ij) + s_j )
	if s==None:
		s = [0]*(A.nrows())
	return max( [ A[i,j].degree() + s[j] for i in range(A.nrows()) for j in range(A.ncols()) if A[i,j].degree() >= 0 ] )

def column_degree( A, s=None ):
	# input: polynomial matrix A and degree shift s
	# output: column degrees of A shifted by s
	if s==None:
		s=[0]*(A.nrows())
	return [max([A[i,j].degree()+s[i] for i in range(A.nrows()) if A[i,j].degree()>=0]) if A[:,j] != 0 else None for j in range(A.ncols())]

def row_degree( A, s=None ):
	# input: polynomial matrix A and degree shift s
	# output: row degrees of A shifted by s
	if s==None:
		s=[0]*(A.ncols())
	return [max([A[i,j].degree()+s[j] for j in range(A.ncols()) if A[i,j].degree()>=0]) if A[i,:] != 0 else None for i in range(A.nrows()) ]

def pivot_info( A, s=None ):
	# input: polynomial matrix A and degree shift s
	# output: s-pivot index and s-pivot degree of A
  if s==None:
    s=[0]*(A.ncols())
  rdeg = row_degree(A,s)
  pivind = [ max( [ j for j in range(A.ncols()) if (A[i,j].degree()+s[j] == rdeg[i]) ] ) for i in range(A.nrows()) ]
  pivdeg = [A[i,pivind[i]].degree() for i in range(A.nrows())]
  return (pivind,pivdeg)

### GENERATING RANDOM MATRICES ###

# random matrix with given column degree
def rand_polmat( baseField, nrows, ncols, deg ):
	# input: base field K, dimensions m and n, maximum degree d
	# output: m x n matrix over K[X] with random entries of degree at most d
	polRing.<X> = baseField[]
	return Matrix(polRing, [ [polRing.random_element(degree=deg) for j in range(ncols)] for i in range(nrows) ] ) 

def rand_polmat_cdeg( baseField, cdeg, nrows=None ):
	# input: base field K, column degree cdeg of length n, dimension m
	# output: m x n matrix over K[X] with random entries and column degree bounded by cdeg componentwise
	# note: if nrows not specified, returns a square matrix
	ncols = len(cdeg)
	if nrows == None:
		nrows = ncols
	polRing.<X> = baseField[]
	M = Matrix( polRing, nrows, ncols )
	for i in range(nrows):
		for j in range(ncols):
			M[i,j] = polRing.random_element( degree=cdeg[j] )
	return M

def rand_polmat_rdeg( baseField, rdeg, ncols=None ):
	# input: base field K, row degree cdeg of length m, dimension n
	# output: m x n matrix over K[X] with random entries and row degree bounded by rdeg componentwise
	# note: if ncols not specified, returns a square matrix
	nrows = len(rdeg)
	if ncols == None:
		ncols = nrows
	polRing.<X> = baseField[]
	M = Matrix( polRing, nrows, ncols )
	for i in range(nrows):
		for j in range(ncols):
			M[i,j] = polRing.random_element( degree=rdeg[i] )
	return M

### GENERATING RANDOM REDUCED AND NORMAL FORMS ###

# random shifted-Popov matrix
# version with increasing pivot indices (square --> pivots on diagonal)
def rand_popov_polmat( baseField, pivdeg, ncols=None, shift=None, details=False ):
	##input: base field K, list 'pivdeg' of length m, number of columns ncols,
	#and a degree shift of length ncols, and details?
	##output: a polynomial matrix over K which is in 'shift'-Popov 
	#form (variant with pivots on the diagonal) with the pivot
	#in row i having degree pivdeg[i]; if details then return the pivot indices
	##requires: ncols >= len(pivdeg)
	##note: if ncols is not specified, returns a square matrix 
	##note: if no shift is specified, it is chosen uniform: shift = [0,...,0]
	##note
	nrows = len(pivdeg)
	if ncols == None:
		ncols = nrows
	if shift==None:
		shift = [0]*ncols
	polRing.<X> = baseField[]
	M = Matrix( polRing, nrows, ncols ) 
	# choose random locations for the pivots among 1...ncols
	pivind = sorted( list(Permutations(ncols).random_element())[:nrows] )
	pivind = [pivind[j]-1 for j in range(len(pivind))]
	for i in range(nrows):
		for j in range(pivind[i]):
			if j in pivind:
				M[i,j] = polRing.random_element( degree=max( -1, min( pivdeg[pivind.index(j)]-1, pivdeg[i]+shift[i]-shift[j] ) ) )
			else:
				M[i,j] = polRing.random_element( degree=max( -1, pivdeg[i]+shift[i]-shift[j] ) )
		M[i,pivind[i]] = X^(pivdeg[i]) + polRing.random_element( degree=pivdeg[i]-1 )
		for j in range(pivind[i]+1,ncols):
			if j in pivind:
				M[i,j] = polRing.random_element( degree=max( -1, min( pivdeg[pivind.index(j)]-1, pivdeg[i]-1+shift[i]-shift[j] ) ) )
			else:
				M[i,j] = polRing.random_element( degree=max( -1, pivdeg[i]+shift[i]-shift[j]-1 ) )
	if details:
		return M, pivind
	else:
		return M

# random shifted weak Popov matrix
# (a bit more specific: increasing pivot indices and monic pivots; but not Popov normalized)
def rand_weak_popov_polmat( baseField, pivdeg, ncols=None, shift=None, details=False ):
	##input: base field K, list 'pivdeg' of length m, number of columns ncols,
	#and a degree shift of length ncols, and details?
	##output: a polynomial matrix over K which is in 'shift'-weak Popov 
	#form with increasing pivot indices and monic pivots, with the pivot
	#in row i having degree pivdeg[i]; if details then return the pivot indices
	##requires: ncols >= len(pivdeg)
	##note: if ncols is not specified, returns a square matrix 
	##note: if no shift is specified, it is chosen uniform: shift = [0,...,0]
	nrows = len(pivdeg)
	if ncols == None:
		ncols = nrows
	if shift==None:
		shift = [0]*ncols
	polRing.<X> = baseField[]
	M = Matrix( polRing, nrows, ncols ) 
	# choose random locations for the pivots among 1...ncols
	pivind = sorted( list(Permutations(ncols).random_element())[:nrows] )
	pivind = [pivind[j]-1 for j in range(len(pivind))]
	for i in range(nrows):
		for j in range(pivind[i]):
			M[i,j] = polRing.random_element( degree=max( -1, pivdeg[i]+shift[i]-shift[j] ) )
		M[i,pivind[i]] = X^(pivdeg[i]) + polRing.random_element( degree=pivdeg[i]-1 )
		for j in range(pivind[i]+1,ncols):
			M[i,j] = polRing.random_element( degree=max( -1, pivdeg[i]-1+shift[i]-shift[j] ) )
	if details:
		return M, pivind
	else:
		return M

def rand_hermite_polmat( baseField, minDegs ):
	# input: diagonal degrees
	# output: random square Hermite form with the given degrees
	m = len(minDegs)
	d = max(minDegs)
	return rand_popov_polmat( baseField, minDegs, shift=[(m-i)*m*d for i in range(m)] ).T

### SHIFTS and REDUCEDNESS ###

def shiftMat( polRing, shift ):
	# Input: polynomial ring, shift *with nonnegative entries*
	# Output: diag( X^shift[i] )
	bF,(X,) = polRing.objgens()
	return Matrix.diagonal( [X^(shift[j]) for j in range(len(shift))] )


def unishift( length, constant=0 ):
	# input: nonnegative number 'length', integer 'constant'
	# output: uniform shift of length 'length' with entries 'constant'
	return [constant] * length

def leading_matrix_row( mat, shift=None ):
	# input: m x n polynomial matrix 'mat', degree shift
	# output: shift-leading matrix of mat (working in rows)
	# assumption: pmat has no zero row
	m = mat.nrows()
	n = mat.ncols()
	lmat = Matrix( mat.base_ring().base_ring(), m, n )
	if shift==None:
		shift = unishift(n)
	for i in range(m):
		# compute shifted row degree
		rdeg = max( [mat[i,j].degree() + shift[j] for j in range(n) if mat[i,j] != 0 ] )
		# row i of lmat = entries of row i of mat achieving this rdeg
		for j in range(n):
			if ( mat[i,j] != 0 and mat[i,j].degree() + shift[j] == rdeg ):
				lmat[i,j] = mat[i,j].leading_coefficient()
	return lmat

def leading_matrix_col( mat, shift=None ):
	# input: m x n polynomial matrix 'mat', degree shift
	# output: shift-leading matrix of mat (working in columns)
	# assumption: pmat has no zero column
	m = mat.nrows()
	n = mat.ncols()
	lmat = Matrix( mat.base_ring().base_ring(), m, n )
	if shift==None:
		shift = unishift(m)
	for j in range(n):
		# compute shifted row degree
		cdeg = max( [mat[i,j].degree() + shift[i] for i in range(m) if mat[i,j] != 0 ] )
		# column j of lmat = entries of column j of mat achieving this cdeg
		for i in range(m):
			if ( mat[i,j] != 0 and mat[i,j].degree() + shift[i] == cdeg ):
				lmat[i,j] = mat[i,j].leading_coefficient()
	return lmat

def is_row_reduced( mat, shift=None ):
	# input: m x n polynomial matrix mat, and column shifts 'shift'
	# output: boolean indicating whether A is shift-row reduced
	# Refs for the non-shifted case:
	# [Gupta et al. 2012 Triangular x-basis decompositions and 
	#derandomization of linear algebra algorithms over K[x], p446]
	# or Kailath 1980
	# note that m > n  implies that mat cannot be row-reduced
	return leading_matrix_row( mat, shift ).rank() == mat.nrows()

def is_column_reduced( mat, shift=None ):
	# input: m x n polynomial matrix mat, and row shifts 'shift'
	# output: boolean indicating whether A is shift-row reduced
	# Refs for the non-shifted case:
	# [Gupta et al. 2012 Triangular x-basis decompositions and 
	#derandomization of linear algebra algorithms over K[x], p446]
	# or Kailath 1980
	# note that m > n  implies that mat cannot be row-reduced
	return leading_matrix_col( mat, shift ).rank() == mat.ncols()

def is_row_popov_form( mat, shift=None ): 
	# input: m x n matrix over K[X], column degree shifts
	# output: boolean indicating whether the matrix is
	#in shift-Popov form (working on the rows)
	if shift==None:
		shift = unishift(mat.ncols())

	popov = True  # will be the output: is mat shift-weak Popov?

	#check the rows are nonzero, have increasing pivot indices, with monic pivots
	i = 0
	pivind = []
	while ( popov and i < mat.nrows() ):
		piv = None  # will be the pivot index of the row
		deg = None  # will be the shift-degree of the row
		for j in range( mat.ncols() ):
			if (mat[i,j] != 0 and mat[i,j].degree() + shift[j] >= deg ):
				piv = j
				deg = mat[i,j].degree() + shift[j]
		 # if the row is zero, the pivot is non-monic, or is not increasing, then not Popov
		if deg==None or mat[i,piv].leading_coefficient()!=1:
			popov = False
		if (i>0 and piv<=max(pivind)):
			popov = False
		pivind.append(piv)
		i += 1
	
	# check the Popov normalization property
	i=0
	while ( popov and i < mat.nrows() ):
		j = pivind[i]
		k=0
		while ( popov and k < mat.nrows() ):
			if (k != i) and (mat[i,j].degree() <= mat[k,j].degree()):
				popov = False
			k = k+1
		i=i+1

	return popov

def is_row_weak_popov_form( mat, shift=None ): 
	# input: m x n matrix over K[X], column degree shifts
	# output: boolean indicating whether the matrix is
	#in shift-weak Popov form (working on the rows)
	if shift==None:
		shift = unishift(mat.ncols())

	wpf = True  # will be the output: is mat shift-weak Popov?
	pivots = set()  # set of pivot indices

	#check the rows are nonzero and have distinct pivot indices
	i = 0
	while ( wpf and i < mat.nrows() ):
		piv = None  # will be the pivot index of the row
		deg = None  # will be the shift-degree of the row
		for j in range( mat.ncols() ):
			if (mat[i,j] != 0 and mat[i,j].degree() + shift[j] >= deg ):
				piv = j
				deg = mat[i,j].degree() + shift[j]
		if deg==None or (piv in pivots): # the row is zero or duplicate pivots
			wpf = False
		else: # record the new pivot index
			pivots.add( piv )
		i += 1

	return wpf

### CONSTANT MATRIX DECOMPOSITIONS ###
### Cubic time algorithms for LSP / LQUP / REF ###
# FIXME Vincent: would be more efficient to use the ones from sage..?! why did I write this, did I encounter some pb?

def LSP( A, verbose=False ):
	# input: matrix A over a field.
	# returns L,S,P,nonzRows such that A = L^-1 S P is
	#the LSP decomposition of A. nonzRows gives the
	#indices of nonzero rows in S.
	m = A.nrows()
	n = A.ncols()
	nonzRows = []
	pivIndex = 0
	L = Matrix.identity(A.base_ring(),m,m)
	S = copy(A)
	P = Matrix.identity(A.base_ring(),n,n)
	for i in range(m):
		#find column with pivot element on row i, if there is some
		pivot = pivIndex
		while (pivot < n and S[i,pivot] == 0):
			pivot += 1
		if pivot < n:
			nonzRows.append(i)
			S.swap_columns(pivIndex,pivot)
			P.swap_columns(pivIndex,pivot)
			for k in range(i+1,m):
				cstFactor = -S[k,pivIndex]/S[i,pivIndex]
				S.add_multiple_of_row(k,i,cstFactor)
				L.add_multiple_of_row(k,i,cstFactor)
			pivIndex += 1
	return L,S,P,nonzRows

def LQUP( A, verbose=False ):
	# returns L,Q,U,P such that A = L^-1 Q U P is the
	# LQUP decomposition of A
	L,U,P,nonzRows = LSP(A)
	m = A.nrows()
	Q = Matrix.identity(A.base_ring(),m,m)
	for i in range(len(nonzRows)):
		U.swap_rows(i,nonzRows[i])
		Q.swap_columns(i,nonzRows[i])
	return L,Q,U,P

def REF( A, passive_rows=set() ):
	# input: matrix A, list of row indices "passive_rows"
	# output: P,L,U row echelon form of A with L P A = U
	# rows with indices in 'passive_rows' are affected by
	#pivoting operations but are never used as a pivot row
	#(default: usual Gaussian eliminations, all rows can be used)
	s_passive_rows = set( passive_rows )
	s_pivot_rows = set( [] ) #rows actually used for pivoting. Note: once in this set a row is no longer affected by pivoting operations
	pivot_rows = [row for row in range(A.nrows()) if row not in s_passive_rows] #rows that can be used for pivoting

	#perm = Permutation( A.nrows() ).identity()
	L = Matrix.identity( A.base_ring(), A.nrows() )
	U = copy( A )
	
	pivCol = 0
	while pivCol < A.ncols() and pivot_rows != []:
		#find pivot in column pivCol
		pivRowPos = 0 # position in pivots_row
		while pivRowPos < len(pivot_rows) and U[pivot_rows[pivRowPos],pivCol] == 0:
			pivRowPos += 1
		if pivRowPos < len(pivot_rows): #else no pivot on this column move to the next one!
			pivRow = pivot_rows[pivRowPos]
			pivot_rows.remove( pivRow )
			s_pivot_rows.add( pivRow )
			# nonzero pivot at pivRow,pivCol
			#FIXME if permuting, should do circular permutation of the rows to keep the order of the pivots...
			#so for now, rather do not permute anything
			for row in s_passive_rows | set(pivot_rows): # perform operations
				scalar = -U[row,pivCol] / U[pivRow,pivCol]
				U.add_multiple_of_row( row, pivRow, scalar )
				L.add_multiple_of_row( row, pivRow, scalar )
		pivCol += 1

	return L,U,s_pivot_rows





#### FIXME Vincent : below probably useless

### CREATE RANDOM INSTANCES ###
def rand_fphps_instance( rdim, cdim, order, baseField ):
	# input
	polRing.<X> = baseField[]
	series = Matrix( polRing, [polRing.random_element(degree=order-1) for i in range(dim)])
	#linseries = Matrix(baseField, [[series[i][j] for j in range(sigma)] for i in range(m)])
	return polRing,series

def rand_koetter_instance( m, sigma, baseField ):
	## FIXME not complete: multiplicities
	polyRing.<X> = baseField[]
	pointsX = vector([baseField.random_element() for i in range(sigma)])
	pointsY = vector([baseField.random_element() for i in range(sigma)])
	matEval = Matrix(baseField, [[pointsY[j]^i for j in range(sigma)] for i in range(m)])
	return polyRing,pointsX,pointsY,matEval

def rand_rs_master( n, baseField, factors=None):
	polRing.<X> = baseField[]
	if ( factors == "master" ):
		G = 1
		a = baseField.primitive_element()
		b=a
		for i in range(n):
			G *= X - b
			b *= a
	elif ( factors == "master_full" ):
		G = X^(baseField.cardinality()-1) - 1
	else:
		G = polRing.random_element(n-1)
		G += X^n
	return G

def rand_gs_polmat( baseField,n,ell,m=1,k=0,masterFactors=None ):
	# input: base field (finite), degree n, lattice dimension ell
	# optional: k to put weights into the matrix (default no weight),
	#masterFactors to give particular shapes to G (default random)
	# output: the lattice matrix L, polynomials R and G
	polRing.<X> = baseField[]
	R = rand_rs_interpolant( n, baseField)
	G = rand_rs_master( n, baseField, masterFactors )
	return guruswami_sudan_polmat( R, G, ell, m, k ),R,G

### BUILD LATTICES AND GENERATORS ###
def guruswami_sudan_polmat( R, G, ell, m=1, k=0 ):
	# input: polynomials G (degree n) and R (degree < n), dimension ell, multiplicity m (default 1, Sudan case) and weight k (default 0, weights not inserted directly into the matrix)
	# output: polynomial lattice matrix for rs list-decoding using Coppersmith technique
	polRing,(X,) = R.parent().objgens()
	L = Matrix( polRing, ell+1 )
	L[0,0] = G**m
	for i in range(1,m+1):
		L[i,0] = -L[i-1,0]*R
		for j in range(1,i+1):
			L[i,j] = X**k * L[i-1,j-1]/G - R * L[i-1,j]/G
	for i in range(m+1,ell+1):
		for j in range(i+1):
			L[i,j] = X**k * L[i-1,j-1]
	return L

### TEST RESULTS OF ALGOS ###
def test_koetter_result( P, pointsX, pointsY, sigma, N=None ):
	#base field, polynomial ring
	baseField = P[0,0].base_ring()
	polyRing.<X> = baseField[]
	polyRing2.<Y> = polyRing[]
	m = P.nrows()

	# test degrees
	degrees = True
	if N == None:
		N = [ceil(sigma/m) for i in range(m)]
	for j in range(m):
		for i in range(m):
			if P[i,j].degree() > N[j]:
				degrees = False

	#test correctness: the m polynomials P_i(X,Y) defined by the rows
	# of P satisfy P_i( x_j, y_j ) = 0 for every j
	correctness = True
	for i in range(m):
		Q = polyRing2(list(P[i]))
		for j in range(sigma):
			if (Q(pointsY[j])(pointsX[j]) != 0):
				correctness = False
	
	return degrees,correctness

def test_hps_result( P, F, sigma, N=None ):
	#base field, polynomial ring
	baseField = P[0,0].base_ring()
	polyRing.<X> = baseField[]
	m = P.nrows()

	# test all rows satisfy the degree constraint
	#FIXME sometimes may want to test if some row satisfies it
	degrees = True
	if N == None:
		N = [ceil(sigma/m) for i in range(m)]
	for j in range(m):
		for i in range(m):
			if P[i,j].degree() > N[j]:
				degrees = False

	#test correctness: the m polynomials P_i(X,Y) defined by the rows
	# of P satisfy P_i( x_j, y_j ) = 0 for every j
	correctness = True
	Res = P * F
	for i in range(m):
		if Res.valuation() < sigma:
			correctness = False
	
	return degrees,correctness

def partial_linearization( A, d=-1, e=[], dA=None ):
	# input: mxm matrix A with degrees in column j
	#bounded by dA[j]
	# output: partially linearized matrix B such as
	#De,d(A) in [Gupta et al., 2012, Theorem 10]
	# if dA is not given, d and e HAVE to be specified
	# if d is not given (or ill-formed), take d = average(dA)
	# if e is not given (or ill-formed), determine it 
	#with respect to the average of dA

	# compute parameters
	bF,(X,) = A.base_ring().objgens()
	m = A.nrows()
	if d < 0:
		if d!=-1:
			print "Warning~~partial_lin~~: invalid degree bound d,"\
					"choosing average of column degrees"
		d = (sum(dA)/m).ceil()
	if len(e) != m:
		if e!=[]:
			print "Warning~~partial_lin~~: invalid e, taking default"
		e = [dA[j]//d for j in range(m)]
	
	AA = Matrix( A.base_ring(), m, m )
	inc = 0 # size of increase (nb of cols and rows added)
	for j in range(m):
		# right now AA is m+inc x m+inc
		# first add e[j] zero rows
		AA = AA.stack( Matrix( A.base_ring(), e[j], m+inc ) )
		# then deal with the low-degree coeffs
		for i in range(m): # does nothing if e[j]==0
			AA[i,j] = A[i,j].truncate(d) #A[i,j]//(X**(e[j]*d))
		if e[j] > 0: AA[m+inc,j] = -X**d
		inc += e[j]
		#if e[j] > 0: AA[m+inc-1,j] = 1 

		# right now AA is m+inc * m+oldinc
		# finally, flatten the high-degree coeffs of column j 
		for ee in range(1,e[j]+1): # does nothing if e[j] == 0
			# move coeffs of degree between ee*d and (ee+1)*d to new column 
			#index of this new column: m+oldinc+ee-1, where oldinc = inc-e[j]
			C = vector( [A[i,j][ee*d:(ee+1)*d]//(X**(ee*d)) for i in range(m)] + [0]*inc )
			AA = AA.augment( C )
			# below, column [0 .. 0 1 -X**d 0 .. 0] with 1 on the diagonal
			AA[m+inc-e[j]+ee-1,m+inc-e[j]+ee-1] = 1
			if ee<e[j]: # no -X**d if this is the last column for this ee
				AA[m+inc-e[j]+ee,m+inc-e[j]+ee-1] = -X**d # just below the 1
	
	return AA

def increasing_rdeg_permutation( rdegs, shift ):
	# input: rdegs are the shift-row degrees of matrix A
	# output: permutation pi such that these shifted
	#row degrees are increasing:
	#(rdegs[pi(i)]+shift[pi(i)])_i is increasing
	pi = map( lambda (a,b): a, sorted( [(i+1,rdegs[i]+shift[i]) for i in range(len(rdegs))], key=lambda (a,b): b) )
	return Permutation( pi )

#### TODO change this to "compress / linearize" functions that are more general ####
def polvec_to_coeffmat( serie, order=None ):
	# input: vector 'serie' of m=serie.degree() polynomials, order to which
	#we will truncate the polynomials in serie
	# output: the m x order matrix whose i,j entry is coeff(F[i],X^j)
	#or 0 when j > min( F[i].degree(), order )
	if order == None: #define it to be the max degree in serie
		order = 1 + max( [serie[i].degree() for i in range(serie.degree())] )
	return Matrix( serie[0].base_ring(), serie.degree(), order, 
			[[serie[i][j] for j in range(order)] for i in range(serie.degree())] )

def coeffmat_to_polvec( coeffSerie, polRing=None ):
	# input: m x sigma matrix coeffSerie
	# output: vector F of polynomials, such that coeff(F[i],X^j) = coeffSerie[i,j]
	if polRing == None:
		polRing.<X> = PolynomialRing( coeffSerie.base_ring() )
	return vector( [ polRing( list(coeffSerie[i])) 
			for i in range(coeffSerie.nrows())] )

def polmat_to_coeffmat( serie, order=None ):
	# input: m x n matrix 'serie' of polynomials, 'order' to which
	#we will truncate the polynomials in 'serie'
	# output: m x n*order matrix whose i,(k*n+j) entry is coeff(F[i,j],X^k)
	#or 0 when j > min( F[i,j].degree(), order )
	#so first n columns are constant coefficients, next n columns are
	#coefficients of X, etc...
	if order == None: #define it to be the max degree in serie
		order = 1 + max( [serie[i,j].degree() 
			for j in range(serie.ncols()) for i in range(serie.nrows())] )
	return Matrix( #serie[0,0].base_ring(), serie.nrows(), order * serie.ncols(), 
			[[serie[i,j][k] for k in range(order) for j in range(serie.ncols())] 
			for i in range(serie.nrows())] )

def coeffmat_to_polmat( coeffSerie, colInd=None, nbCols=None, polRing=None ):
	# colInd from 0 to maximum column
	if polRing == None:
		polRing = PolynomialRing( coeffSerie.base_ring(), 'X' )
	(X,) = polRing.objgens()[1]
	if colInd == None:
		assert( nbCols != None )
	if nbCols == None: # nbCols columns indexed 0...nbCols-1
		nbCols = max(colInd)+1
	curDeg = [0] * coeffSerie.nrows()
	polMat = Matrix( polRing, coeffSerie.nrows(), nbCols )
	for j in range( coeffSerie.ncols() ):
		for i in range( coeffSerie.nrows() ):
			polMat[i,colInd[j]] += X^(curDeg[colInd[j]]) * coeffSerie[i,j]
		curDeg[colInd[j]] += 1
	return polMat

def points_to_vdm( points, m=None ):
	# input: vector of points x1,...,xn in some ring, integer m
	# output: rectangular Vandermonde matrix with coeff[i,j] = xj^i for j=1..n and i=0..m-1
	# default value for m: m = n such that the matrix is square
	if m==None:
		m=points.degree()
	return Matrix(points.base_ring(),m,[[points[j]^i for j in range(points.degree())] for i in range(m)])
