def write_mat(fname, mats, p, M):
	f = open(fname,'w')
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

def setup(n, p, deg, terms, D):
	global mul_mats,fs
	mul_mats = []
	fs = []
	for i in range(n):
		fs.append(M1.random_element(deg,terms))
	ideal = Ideal(fs)
	init(ideal,GF(p))
	write_mat('../linbox/test.dat', mul_mats, p, D)
