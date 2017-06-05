matrix_bm = True
appbas = False
popov_appbas = False
unbalanced_appbas = False
unbalanced_popov_appbas = False

### LOAD THE MAIN CODE ###
load('matrix-berlekamp-massey.sage')

### DEFINE SOME USEFUL TOOLS ### 
def degree_matrix( A ): #matrix of degrees for a polynomial matrix
	return Matrix( ZZ, [[A[i,j].degree() for j in range(A.ncols())] for i in range(A.nrows())] )

def valuation_matrix( A ): #matrix of valuations for a polynomial matrix
	return Matrix( ZZ, [[A[i,j].valuation() if A[i,j].valuation() != +Infinity else -1 for j in range(A.ncols()) ] for i in range(A.nrows())] )

def is_row_reduced( mat, shift=None ):
	# input: m x n polynomial matrix mat, and shift 'shift'
	# output: boolean indicating whether A is shift-row reduced
	return leading_matrix_row( mat, shift ).rank() == mat.nrows()

def is_row_popov_form( mat, shift=None ): 
	# input: m x n matrix over K[X], shift
	# output: boolean indicating whether the matrix is
	#in shift-Popov form
	if shift==None:
		shift = [0]*(mat.ncols())

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

### EXAMPLES ### 

if matrix_bm:
	bF = GF(97)
	# initialize recurrent sequence as U T^k V, k = 0,...,deg
	D = 256 # dimension of the multiplication matrix T
	m = 16 # row dimension of U
	n = 8 # column dimension of V
	nterms = RR((m+n)*D/(m*n)).ceiling()  # expected number of terms we need
	print "Input sequence of the form U T^k V, for matrices U: mxD, T: DxD, V: Dxn"
	print "Dimensions m , D , n : ",m,",",D,",",n
	print "Chosen number of terms: ceiling of (m+n)D / mn  -->  ",nterms
	print "(should be sufficient with high probability)"
	T = Matrix.random(bF,D)
	U = Matrix.random(bF,m,D)
	V = Matrix.random(bF,D,n)
	TkV = copy(V)  # stores U*T^k*V, here for k=0
	seq = []  # stores the sequence
	for k in range(nterms):
		seq.append( U*TkV )
		TkV = T*TkV
	mingen = matrix_berlekamp_massey(seq)
	print "Popov basis computed. Degrees:\n",degree_matrix(mingen)
	longnterms = 2*D+1
	longTkV = copy(V)  # stores U*T^k*V, here for k=0
	longseq = []  # stores the sequence
	for k in range(longnterms):
		longseq.append( U*longTkV )
		longTkV = T*longTkV
	print "To check the result, we now compute a basis for more terms: ",longnterms
	longmingen = matrix_berlekamp_massey(longseq)
	print "... done. Did we obtain the same basis as for fewer terms? --> ",longmingen==mingen
	shorter = 1
	print "To check if bound not too pessimistic, let's compute a basis for less terms: ",nterms-shorter
	shortmingen = matrix_berlekamp_massey(seq[:nterms-shorter])
	print "... done. Did we obtain the same basis? --> ",shortmingen==mingen

if appbas:
	print "##############################" 
	print "### EXAMPLE: APPROXIMANT BASIS COMPUTATION"
	print "##############################" 
	bF = GF(257)
	pR.<X> = bF[]
	m = 8
	n = 4
	order = 128
	serie = Matrix.random(pR,m,n,degree=order-1)
	approx = iter_appbas( serie, order )
	print "degrees in minimal basis:\n",degree_matrix(approx)
	print "valuations in residual:\n",valuation_matrix(approx*serie)
	print "is minimal basis indeed minimal? --> ",is_row_reduced(approx)

if popov_appbas:
	print "##############################" 
	print "### EXAMPLE: POPOV APPROXIMANT BASIS COMPUTATION"
	print "##############################" 
	bF = GF(257)
	pR.<X> = bF[]
	m = 8
	n = 4
	order = 128
	serie = Matrix.random(pR,m,n,degree=order-1)
	approx = iter_popov_appbas( serie, order )
	print "degrees in minimal basis:\n",degree_matrix(approx)
	print "valuations in residual:\n",valuation_matrix(approx*serie)
	print "is minimal basis indeed minimal? --> ",is_row_reduced(approx)
	print "is minimal basis indeed in Popov form? --> ",is_row_popov_form(approx)


if unbalanced_appbas:
	print "##############################" 
	print "### EXAMPLE: APPROXIMANT BASIS COMPUTATION (LARGE DEGREE OUTPUT)" 
	print "##############################" 
	bF = GF(257)
	pR.<X> = bF[]
	m = 8
	order = 128
	serie = Matrix.random(pR,m,1,degree=order-1)
	serie[0,0] = sum([X^i for i in range(order)])
	serie[1,0] = 1 + sum([2*X^i for i in range(order)])
	for i in range(2,m/2):
		serie[i,0] = X * serie[i-1,0]
	print "after m/2 steps, many constant polynomials in the approx\n", \
	degree_matrix(iter_appbas(serie,m/2,[0]*4+[order]*4)).str()
	approx = iter_appbas(serie,order,[0]*4+[order]*4)
	print "after order steps, many polynomials of degree around order - m/2\n", \
	degree_matrix(approx).str()
	print "(note that here the shift is not uniform: shift =", [0]*4+[order]*4,")"

if unbalanced_popov_appbas:
	print "##############################" 
	print "### EXAMPLE: APPROXIMANT BASIS COMPUTATION (LARGE DEGREE OUTPUT)" 
	print "##############################" 
	bF = GF(257)
	pR.<X> = bF[]
	m = 8
	order = 128
	serie = Matrix.random(pR,m,1,degree=order-1)
	serie[0,0] = sum([X^i for i in range(order)])
	serie[1,0] = 1 + sum([2*X^i for i in range(order)])
	for i in range(2,m/2):
		serie[i,0] = X * serie[i-1,0]
	print "after m/2 steps, many constant polynomials in the approx\n", \
	degree_matrix(iter_appbas(serie,m/2,[0]*4+[order]*4)).str()
	approx = iter_appbas(serie,order,[0]*4+[order]*4)
	print "after order steps, many polynomials of degree around order - m/2\n", \
	degree_matrix(approx).str()
	print "(note that here the shift is not uniform: shift =", [0]*4+[order]*4,")"
	print "now in the s-Popov basis, the column degrees are small on average:"
	approx = iter_popov_appbas(serie,order,[0]*4+[order]*4)
	print degree_matrix(approx).str()

