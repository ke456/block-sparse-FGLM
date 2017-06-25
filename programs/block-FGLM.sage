# provides functions for block-FGLM

def find_lex_basis(ideal,field,M):
	load("poly-solv.sage")
	load("polmat/utils.sage")
	load("polmat/approximant-basis.sage")
	load("polmat/recurrence-generator.sage")
	load("create_block_matrix.sage")
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
	Spair = matrix_berlekamp_massey_reverse(s1)
	S = Spair[0]
	# finding P -the minimal polynomial of T1
	P_bar = S.det()
	P = P_bar.reverse()
	# find the numerator
	N1 = Spair[1]
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
		index = i
		Ti = mul_mats[i]
		Z = [ U* T1^i * Ti * V* X^i for i in range(d)] # probably should be stored...
		Z = add(Z)
		N = (S*Z) % X^(d)
		f = vector([N[i,0] for i in range(N.nrows())])
		n_bar = (u*f)[N.nrows()-1] % P_bar
		n = n_bar.reverse()
		func = (n*n1.inverse_mod(P))%P
		print(func)
		mX = 0
		for i in range(len(func.coefficients())):
			mX += func.coefficients()[i] * m^i
		result.append(monomials[index] - mX)
	return result
	
	
	
	
	
	
	
