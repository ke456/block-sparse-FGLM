def solve_wiedemann(B,t):
	v = vector([randint(0,1000) for i in range(B.nrows())])
	u = vector([randint(0,1000) for i in range(B.nrows())])
	b = B*v
	a = [u * B^i * b for i in range(2 * B.nrows())]
	P = berlekamp_massey(a)
	p_0 = P[0]
	w = 0
	for i in range(P.degree()):
		w += P[i+1] * B^i * t
	w = (-1/p_0) * w
	return w
