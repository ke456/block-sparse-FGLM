def find_func(U, V, T1, S, u, P, P_bar, n1, Tn,D, M):
	Z = [ U* T1^i * Tn * V* t^i for i in range(D/M)]
	Z = add(Z)
	N = (S*Z) % t^(D/M)
	f = vector([N[i,0] for i in range(N.nrows())])
	n_bar = (u*f)[N.nrows()-1] % P_bar
	n = n_bar.reverse()
	print(n)
	return (n*n1.inverse_mod(P))%P
