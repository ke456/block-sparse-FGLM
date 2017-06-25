#### Approximant basis algorithms

##Definition (approximant basis)
# Given:
#   * m x n matrix of univariate polynomials 'pmat',
#   * approximation order 'order' (list of n positive integers),
# An approximant basis for (pmat,order) is a matrix over K[X]
# whose rows form a basis for the K[X]-module
# { 'app' in K[X]^{1 x m}  |  the column j of 'app' 'pmat' is 0 modulo X^{order[j]} }

## Minimal and Popov approximant bases
# Given in addition:
#   * a degree shift 'shifts' (list of m integers)
# then an approximant basis for (pmat,order) is said to be
# "a shift-minimal" (resp. "the shift-Popov") approximant basis
# if it shift-reduced (resp. in shift-Popov form)
# Idem for shift-ordered weak Popov
# Cf. literature for definitions

###############################
## MINIMAL APPROXIMANT BASIS ##
###############################

def iter_appbas( pmat, order, shift=None, iteration="degree" ):
	##Input:
	#   * m x n matrix of polynomials 'pmat',
	#   * approximation order 'order' (list of n positive integers),
	#   * degree shift 'shifts' (list of m integers)
	#   * option 'iteration' specifying how to iterate (TODO describe, see below)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output:
	#   * a shift-ordered weak Popov approximant basis 'appbas'
	#     for '(pmat,order)'
	#   * the shift-row degree 'rdeg' of 'appbas'
	##References for this kind of iterative approximant basis algorithms:
	# [van Barel-Bultheel, 1992, Numer. Algor.]
	# [Beckermann-Labahn, 1994, SIAM J. Matrix Anal. & Appl.]
	##Here, the iteration is by increasing order: first all columns
	# modulo X, then all columns modulo X^2, etc.
	
	# Define parameters and perform some sanity checks
	m = pmat.nrows()
	n = pmat.ncols()
	polRing,(X,) = pmat.base_ring().objgens()

	# Define 'restorder' : will be updated with the order that remains to be dealt with
	# Define 'restindex' : indices of orders that remains to be dealt with
	restorder = copy(order)
	restindex = range(n)

	# Define the shift-row degrees 'rdeg' that will be updated throughout the algorithm
	# --> 'rdeg' is initially the shift-row degree of the identity matrix
	# recall default = uniform shift --> rdeg = 0-row degree of Id = [0,...,0]
	rdeg = copy(shift) if shift != None else [0]*m

	# initialization of the residuals and the approximant basis
	appbas = Matrix.identity( polRing, m )
	residuals = copy(pmat)

	while len(restorder)>0:
		# invariant:
		#   * appbas is a shift-ordered weak Popov approximant basis for (pmat,doneorder)
		#   where doneorder = ... TODO + restorder == order (componentwise)
		#   * rdeg is the shift-row degree of appbas
		#   * residuals = submatrix of columns (appbas * pmat)[:,j] for all j such that restorder[j] > 0

		# currently two choices for next coefficient to deal with:
		#   * first of the largest entries in order (--> process 'pmat' degreewise, and left to right)
		#   * first one in order (--> process 'pmat' columnwise, from left column to right column)
		if iteration == "degree":
			j = min( [ind for ind in range(len(restorder)) if restorder[ind] == max(restorder)] )
		elif iteration == "column":
			j = 0
		else:
			raise ValueError("Input value for option 'iteration' should be 'degree' or 'column'.")
		d = order[restindex[j]] - restorder[j]

		# the coefficient of degree d of the column j of residual
		# --> this is very likely nonzero and we want to make it zero, so that this column becomes zero mod X^{d+1}
		constRes = vector( [residuals[i,j][d] for i in range(m)] )

		#Lambda: collect rows with nonzero coefficient
		#pi: index of the first row with smallest shift (among those in Lambda)
		Lambda = []
		pi = -1
		for i in range(m):
			if constRes[i] != 0:
				Lambda.append(i)
				if pi < 0 or rdeg[i] < rdeg[pi]:
					pi = i
		if Lambda != [] : # otherwise nothing to do
			#update all rows in Lambda--{pi}
			Lambda.remove(pi)
			for row in Lambda:
				scalar = -constRes[row]/constRes[pi]
				appbas.add_multiple_of_row( row, pi, scalar )
				residuals.add_multiple_of_row( row, pi, scalar )
			#update row pi
			rdeg[pi] += 1
			appbas.rescale_row( pi, X )
			residuals.rescale_row( pi, X )

		# Decrement restorder[j],
		# unless there is no more work to do in this column, i.e. if restorder[j] was 1:
		# in this case remove the column j of residual,restorder,restindex
		if restorder[j]==1:
			residuals = residuals.delete_columns( [j] )
			restorder.pop(j)
			restindex.pop(j)
		else:
			restorder[j] -= 1
	return appbas,rdeg

def uniorderone_appbas( mat, shift=None ):
	##Input:
	#   * m x n matrix of field elements 'mat',
	#   * degree shift 'shifts' (list of m integers)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output: 'appbas' = the shift-Popov approximant basis
	#         for ('mat',[1,...,1])
	# References:
	#  * [Jeannerod-Neiger-Schost-Villard, JSC 2017, Section 7] for a more general case
	#  * [Jeannerod-Neiger-Schost-Villard, preprint fastpab] for this specific case
	m = mat.nrows()
	rdeg = copy(shift) if shift != None else [0]*m
	# Find the permutation which stable-sorts the shift in nondecreasing order
	perm = Permutation(list(zip(*sorted( [(rdeg[i],i+1) for i in range(m)] ))[1]))
	# Permute the rows of mat accordingly
	matt = mat.with_permuted_rows(perm)
	kermat = matt.kernel(basis="pivot").matrix() # FIXME not sure 'basis=pivot' computes what is intended in Sage, but as of now (June 16 2017) this does precisely what I need
	pivind = []  # the kth element in this list will be the pivot index in row k of kermat
	for i in range(kermat.nrows()):
		j = kermat.ncols()-1
		while j>=0 and kermat[i,j] == 0:
			j -= 1
		pivind.append(j) # note that there must be a nonzero coefficient in kermat[i,:]
	pR.<X> = mat.base_ring()[]
	appbas = Matrix( pR, m, m )
	for i in range(m):
		try:
			ind = pivind.index(i)
			appbas[i,:] = copy(kermat[ind,:])
		except ValueError:
			appbas[i,i] = X
			rdeg[i] += 1
	# permute back the matrix
	appbas.permute_rows_and_columns(perm.inverse(),perm.inverse())
	return appbas,rdeg

def uniorder_dnc_appbas( pmat, order, shift=None ):
	##Input:
	#   * m x n matrix of polynomials 'pmat',
	#   * approximation order 'order': a single positive integer,
	#   * degree shift 'shifts' (list of m integers)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output:
	#   * a shift-ordered weak Popov approximant basis 'appbas'
	#     for '(pmat,[order,...,order])'
	#   * the shift-row degree 'rdeg' of 'appbas'
	if order==1:
		return uniorderone_appbas(pmat(0),shift)
	else:
		halforder = order//2
		app1,rdeg1 = uniorder_dnc_appbas( matrix_truncate(pmat,halforder), halforder, shift )
		residual = matrix_truncate( matrix_shift( app1 * pmat, -halforder), order-halforder )
		app2,rdeg2 = uniorder_dnc_appbas( residual, order-halforder, rdeg1 )
		return app2*app1, rdeg2

def minimal_approximant_basis( serie, order, shift=None, algorithm="iter" ):
	# FIXME spec?
	# define m and n
	if shift != None and len(shift) != m:
		raise ValueError("The shift list does not have the right length -- it should coincide with the number of rows in the input matrix.")
	if n == 1 and type(order) == Integer:
		properorder = [order]
	elif n > 1 and type(order) == Integer:
		print "The order seems to be a single integer (call it 'd'), I was expecting a list..."
		properorder = [order]*n
		print "... order converted to the list [d,...,d]"
	elif type(order) == list and len(order) != n:
		raise ValueError("The order list does not have the right length -- it should coincide with the number of columns in the input matrix.")
	else:
		properorder = copy(order)
	if algorithm=="iter":
		return iter_appbas( serie, properorder, shift )
	#elif algorithm=="mbasis":
	#	return mbasis( serie, order, shift )
	#elif algorithm=="pmbasis":
	#	print "pmbasis still has to be checked"
	#	return 0
	else:
		return iter_appbas( serie, properorder, shift )

#############################
## POPOV APPROXIMANT BASIS ##
#############################

def popov_iter_appbas( serie, order, shift=None ):
	##Input:
	#   * m x n matrix of polynomials 'serie',
	#   * an integer 'order',
	#   * degree shift 'shifts' (list of m integers)
	##Default shift==None: shift will be set to the uniform [0,..,0]
	##Output: the unique approximant basis 'appbas' for 'serie'
	#at order X^order which is in shift-Popov form
	appbas,rdeg = iter_appbas( serie, order, shift )
	mindeg = [rdeg[i] - shift[i] for i in range(appbas.nrows())]
	reduced = iter_appbas( serie, order, [-mindeg[i] for i in range(len(mindeg))] )[0]
	return leading_matrix_row( reduced, [-mindeg[i] for i in range(len(mindeg))] ).I * reduced

def popov_approximant_basis( serie, order, shift=None, algorithm="iter" ):
	# FIXME spec?
	if algorithm=="iter":
		return iter_popov_appbas( serie, order, shift )
	#elif algorithm=="mbasis":
	#	return mbasis( serie, order, shift )
	#elif algorithm=="pmbasis":
	#	print "pmbasis still has to be checked"
	#	return 0
	else:
		return iter_popov_appbas( serie, order, shift )
