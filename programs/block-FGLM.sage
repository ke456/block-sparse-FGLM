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
	print("U:")
	print(U)
	V = Matrix(field,[[randint(1,10000) for i in range(D)] for j in range(M)]).transpose()
	print("V:")
	print(V)
	d = ceil(D/M)
	result = []
	# finding the minimum generating matrix
	T1 = mul_mats[0]
	s1 = [U*T1^i*V for i in range(2*d)] # this is horribly inefficient...
	print("s1:")
	print(s1)
	Spair = matrix_berlekamp_massey(s1)
	S = Spair[0]
	print("S:")
	print(S)
	# find the numerator
	Z1 = add([s1[i].columns()[0] * X^i for i in range(d)])
	Z1 = vec_reverse(Z1,d,M)
	N = S*Z1
	N = shift(N,d,M)
	smith = S.smith_form()
	inv_factors = smith[0].diagonal();
	# finding P -the minimal polynomial of T1
	P = inv_factors[M-1]
	u = smith[1];
	v = smith[2];
	w = v.rows()[M-1]
	w = P * w
	for i in range(M):
		w[i] = w[i] / inv_factors[i]
	u_tilde = w*u
	n1 = (u_tilde*N)
	print("n1:")
	print(n1)
	# adding P to the result
	P = P.squarefree_decomposition().radical_value()
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
		n = (u_tilde*N)
		print("n:")
		print(n)
		func = (n*n1.inverse_mod(P))%P
		mX = 0
		for i in range(len(func.coefficients(false))):
			mX += func.coefficients(false)[i] * m^i
		result.append(monomials[index] - mX)
	return result
	
	
	
	
	
	
	
