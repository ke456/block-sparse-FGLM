# find the function for the Xi (given Ti) using Blocks
def find_func(U, V, T1, S, u, P, P_bar, n1, Ti,D, M):
	Z = [ U* T1^i * Ti * V* X^i for i in range(ceil(D/M))]
	Z = add(Z)
	N = (S*Z) % X^(ceil(D/M))
	f = vector([N[i,0] for i in range(N.nrows())])
	n_bar = (u*f)[N.nrows()-1] % P_bar
	n = n_bar.reverse()
	print(n)
	return (n*n1.inverse_mod(P))%P
