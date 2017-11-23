load("poly-solv.sage")
def write_mat(fname, mats, p, M):
	f = open(fname,'w')
	f.write("Test{}\n".format(mats[0].nrows()))
	f.write(str(p))
	f.write("\n")
	f.write(str(len(mats)))
	f.write("\n")
	f.write(str(M))
	f.write("\n")
	for mat in mats:
		for i in range(mat.nrows()):
			for j in range(mat.ncols()):
				if (mat[i,j] != 0):
					f.write(str(i))
					f.write(" ")
					f.write(str(j))
					f.write(" ")
					f.write(str(mat[i,j]))
					f.write("\n")
		D = mat.nrows()
		f.write(str(D))
		f.write(" ")
		f.write(str(D))
		f.write(" ")
		f.write(str(D))
		f.write("\n")

fs = []

def setup(p, deg, terms):
	n = 3
	D = deg^n
	global mul_mats,fs
	mul_mats = []
	fs = []
	M1.<x1,x2,x3> = PolynomialRing(GF(p))
	for i in range(n):
		fs.append(M1.random_element(deg,terms))
	ideal = Ideal(fs)
	init(ideal,GF(p))
	write_mat('data/test{}.dat'.format(D), mul_mats, p, D)
	f = open('data/test{}.sage'.format(D), 'w')
	f.write("def eval(x1,x2,x3):\n\t[")
	for i in range(len(fs)):
		f.write("{}".format(fs[i]))
		if (i != len(fs) -1):
			f.write(",")
	f.write("]\n")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
