# provides functions for block-FGLM
def vec_reverse(vec,deg,l):
	return vector([vec[i].reverse(deg) for i in range(l)])
	
def shift(vec,deg,l):
	return vector([(vec[i].reverse() %X^deg).reverse() for i in range(l)])

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
	Spair = matrix_berlekamp_massey(s1)
	S = Spair[0]
	# finding P -the minimal polynomial of T1
	P = S.det()
	# find the numerator
	Z1 = add([s1[i].columns()[0] * X^i for i in range(d)])
	Z1 = vec_reverse(Z1,d,M)
	N = S*Z1
	N = shift(N,d,M)
	u = S.smith_form()[1]
	n1 = (u*N)[N.length()-1]
	# adding P to the result
	monomials = ideal.random_element(1).parent().gens()
	m = monomials[0]
	mX = 0
	for i in range(len(P.coefficients(false))):
		mX += P.coefficients(false)[i] * m^i
	result.append(mX)
	# finding rest of the functions
	for i in [1 .. len(mul_mats)-1]:
		index = i
		Ti = mul_mats[i]
		Z = add([ (U* T1^i * Ti * V.columns()[0]* X^i) for i in range(d)]) # probably should be stored...
		Z = vec_reverse(Z,d,M)
		N = (S*Z)
		N = shift(N,d,M)
		n = (u*N)[N.length()-1]
		func = (n*n1.inverse_mod(P))%P
		mX = 0
		for i in range(len(func.coefficients(false))):
			mX += func.coefficients(false)[i] * m^i
		result.append(monomials[index] - mX)
	return result
	
	
	
	
	
	
	
