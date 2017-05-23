M.<x1,x2,x3>=PolynomialRing(GF(11),3,order='degrevlex') 
ideal=Ideal([x1^2 + 2*x2 - 2, x1*x2 + 2*x1 + 4*x2 + 4, x2^2 + 2*x1 - 2*x2 - 5, x3 - 2])
G=ideal.groebner_basis() 
B=ideal.normal_basis()
L=ideal.minimal_associated_primes() 
L = [ ell.groebner_basis() for ell in L]

M.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12>= PolynomialRing(GF(9001),12)

M.<u00,u10,u20,u30,u01,u11,u21,u31,v00,v10,v20,v30,v01,v11,v21,v31,a,b,c,d>=PolynomialRing(GF(9001))
