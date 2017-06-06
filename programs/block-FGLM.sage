# provides functions for block-FGLM
load("poly-solv.sage")
load("matrix-berlekamp-massey.sage")
load("create_block_matrix.sage")

def find_lex_basis(ideal,field,M):
	global mul_mats, B
	Mx.<X> = PolynomialRing(field)
	init(ideal,field)
	D = len(B)
	U = Matrix(field,[[randint(1,10000) for i in range(D)] for j in range(M)])
	V = Matrix(field,[[randint(1,10000) for i in range(D)] for j in range(M)]).transpose()
	d = ceil(D/M)
	result = []
	# finding the minimum generating matrix
	T1 = mul_mats[0]
	s1 = [U*T1^i*V for i in range(2*d)] # this is horribly inefficient...
	S = matrix_reverse(matrix_berlekamp_massey(s1), d) # only a temporary fix, may not reduce
	# finding P -the minimal polynomial of T1
	P_bar = S.det()
	P = P_bar.reverse()
	# find the numerator
	Z1 = add([s1[i]*X^i for i in range(len(s1))])
	N1 = (S*Z1) % X^d
	u = S.smith_form()[1]
	f = vector([N1[i,0] for i in range(M)])
	n1_bar = (u*f)[M-1]
	n1 = n1_bar.reverse()
	# adding P to the result
	monomials = ideal.random_element(1).parent().gens()
	m = monomials[0]
	mX = 0
	for i in range(len(P.coefficients())):
		mX += P.coefficients()[i] * m^i
	result.append(mX)
	# finding rest of the functions
	for i in [1 .. len(mul_mats)-1]:
		Ti = mul_mats[i]
		Z = [ U* T1^i * Ti * V* X^i for i in range(ceil(D/M))] # probably should be stored...
		Z = add(Z)
		N = (S*Z) % X^(ceil(D/M))
		f = vector([N[i,0] for i in range(N.nrows())])
		n_bar = (u*f)[N.nrows()-1] % P_bar
		n = n_bar.reverse()
		func = (n*n1.inverse_mod(P))%P
		print(func)
		mX = 0
		for i in range(len(func.coefficients())):
			mX += func.coefficients()[i] * m^i
		print("finished")
		result.append(mX)
	return result
	
	
	
	
	
	
	
