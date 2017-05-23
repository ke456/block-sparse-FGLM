def create_block_hankel(lst, rows, cols):
	m = []
	for i in range(rows):
		r = lst[i]
		for j in [1 .. cols-1]:
			r = r.augment(lst[i+j])
		m.append(r)
	M = m[0].transpose()
	for i in [1 .. len(m)-1]:
		M = M.augment(m[i].transpose())
	return M.transpose()
	
def create_block_toeplitz(lst, rows, cols):
	m = []
	cur = [lst[i] for i in [0 .. cols-1]]
	for i in range(rows):
		r = cur[0]
		for j in [1 .. cols-1]:
			r = r.augment(cur[j])
		m.append(r)
		if (cols + (i) < len(lst)):
			cur = [lst[cols+(i)]] + cur
	M = m[0].transpose()
	for i in [1 .. len(m)-1]:
		M = M.augment(m[i].transpose())
	return M.transpose()
