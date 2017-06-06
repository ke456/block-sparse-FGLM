# Provides some functions for initialization (such as creating the multiplication matrices)
# as well as converting to lex basis via sparse FGLM (non-blocked)

var('I') # ideal
var('B') # monomial basis
var('G') # Groebner basis
var('l') # linear mapping
var('P','P_bar','N','N_bar')
var('W','MX')
mul_mats = []
mat_dict = []
G2 = []
G_one_var = []

# This creates the multiplication matrices and stores in mul_mats and mat_dict
# Needs to be called first before using any of the other functions
def init(ideal, field):
  global I,B,G,l,W,MX
  W.<x>=PolynomialRing(field)
  MX.<X> = PolynomialRing(field)
  I = ideal
  G = I.groebner_basis()
  B = I.normal_basis()
  for i in I.random_element(1).parent().gens():
    mat = create_mult_mat(i)
    mat_dict.append([i,mat])
    mul_mats.append(mat)
  print(mul_mats)
  l = vector([randint(1,1000) for i in range(B.nmonomials())])
  

# Given the monomial, it creates the multiplication matrix for it
def create_mult_mat(monomial):
  l = []
  for i in B:
    f = I.reduce(monomial * i)
    q = []
    for j in B:
      q.append(f[j])
    l.append(q)
  return Matrix(l).transpose()
  
# finds P, the minimal polynomial, of Ti, where i = index, as well Ni such that
# P*Ni = Z, Z = sum(s[i] * x^i)  
def find_P(index=0):
	global P,P_bar,N,N_bar,G2
	m = mul_mats[0]
	s = [l*vector((m^i).transpose()[m.nrows()-1]) for i in range(2 * B.nmonomials())]
	print(s)
	P = (W)(berlekamp_massey(s))
	P_bar = P.reverse()
	S = (P.parent())(s)
	N_bar = (P.parent())((P_bar * S) % (P.parent())(x^P.degree()))
	N = (N_bar.reverse())
	L = 0
	var = mat_dict[0][0]
	for i in range(P.degree()+1):
		L += P.list()[i] * var^i
	G2.append(L)
	
# computes and stores the rest of the functions for the Groebner basis with respect to
# lex ordering
def find_rest():
  global P,P_bar,N,N_bar,index_used,G2
  find_P()
  while (P.degree() < B.nmonomials()):
    l = vector([randint(1,1000) for i in range(B.nmonomials())])
    find_P()
  m = mul_mats[0]
  var = mat_dict[0][0]
  n = I.random_element(1).parent().ngens()
  for i in [1 .. n-1]:
  	print("start")
  	c_i = vector(mul_mats[i].transpose()[m.nrows()-1])
  	s = [l*m^j*c_i for j in range(B.nmonomials())]
  	S = (P.parent())(s)
  	N_ibar = (P.parent())((P_bar * S) % (P.parent())(x^P.degree()))
  	N_i = N_ibar.reverse()
  	G_i = (N_i * N.inverse_mod(P)) % P
  	print(G_i)
  	L = 0
  	v = mat_dict[i][0]
  	for j in range(G_i.degree()+1):
  		L += G_i.list()[j] * var^j
  	G_one_var.append(L)
  	G2.append(v - L)






















	

