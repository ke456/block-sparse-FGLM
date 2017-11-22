cd ~/repository/polmat/
ls
cd sage/
ls
clear
ls
%runfile init.sage
pR.<X> = GF(97)[]
M = Matrix.random(pR,4,4,degree=64)
S = M.smith_form()
S
degree_matrix(S)
degree_matrix(S[0])
degree_matrix(S[1])
degree_matrix(S[2])
K = minimal_kernel_basis(F[:,:3])
K = minimal_kernel_basis(M[:,:3])
K = minimal_kernel_basis(M[:,:3],[0]^4)
K = minimal_kernel_basis(M[:,:3],[0]*4)
K = minimal_kernel_basis_naive(M[:,:3],[0]*4)
%runfile init.sage
K = minimal_kernel_basis_naive(M[:,:3],[0]*4)
%runfile init.sage
%runfile init.sage
K = minimal_kernel_basis_naive(M[:,:3],[0]*4)
degree_matrix(K)
degree_matrix(K[0])
degree_matrix(K[0]*M)
K[0]*M[:,3] == S[0][3]
K[0]*M[:,3]
S[0][3]
S[0][3,3]
S[0][3,3]/91
K[0]*M[:,3] /71
s = [pR.random_element(4)]
s.append( s[0]*pR.random_element(2)^2)
s.append( s[1]*pR.random_element(3))
s.append( s[2]*pR.random_element(6)^3)
s
s[0] = s[0]/24
s[1] = s[1]/89
s[2] = s[2]/33
s[3] = s[3]/30
s
S = Matrix.diagonal(s)
S
degree_matrix(S)
U = Matrix.random(pR,4,4,"unimodular")
degree_matrix(U)
U = U*Matrix.random(pR,4,4,"unimodular")
U = U.T*Matrix.random(pR,4,4,"unimodular").T
degree_matrix(U)
V = Matrix.random(pR,4,4,"unimodular").T
V = V*Matrix.random(pR,4,4,"unimodular")
V = V*Matrix.random(pR,4,4,"unimodular")*V
degree_matrix(V)
M = U*S*V
degree_matrix(V)
SS = M.smith_form()
SS[0][0,0]
s[0]
[SS[0][i,i] for i in range(4)] == s
SS[1] == U
degree_matrix(SS[1])
%history
K = minimal_kernel_basis_naive(M[:,:3],[0]*4)
degree_matrix(K*M)
degree_matrix(K[0]*M)
K[0]*M[:,3]
s[3]*96
K2 = minimal_kernel_basis_naive(M[:,1:],[0]*4)
K[0]*M[:,0]
K2[0]*M[:,0]
K2[0]*M[:,0]/49 == s[3]

