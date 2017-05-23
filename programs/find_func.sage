def find_func(U, V, T1, S, u, P, P_bar, n1, Tn):
	Z = [ U* T1^i * Tn * V* t^i for i in range(4)]
	Z = add(Z)
	N = (S*Z) % t^T1.nrows()
	f = vector([N[i,0] for i in range(N.nrows())])
	n_bar = (u*f)[N.nrows()-1] % P_bar
	n = n_bar.reverse()
	return (n*n1.inverse_mod(P))%P
