sage: pR.<X> = GF(97)[]
sage: s = [pR.random_element(4)]
sage: s.append( s[0]*pR.random_element(2)^2)
sage: s.append( s[1]*pR.random_element(3))
sage: s.append( s[2]*pR.random_element(6)^3)
sage: s

[57*X^4 + 8*X^3 + 3*X^2 + 67*X + 39,
 5*X^8 + 55*X^7 + 54*X^6 + 60*X^5 + 36*X^4 + 70*X^3 + 32*X^2 + 3*X + 60,
 85*X^11 + 84*X^10 + 77*X^9 + 51*X^8 + 56*X^7 + 79*X^6 + 29*X^5 + 32*X^4 + 63*X^3 + 14*X^2 + 51*X + 10,
 47*X^29 + 16*X^28 + 93*X^27 + 80*X^26 + 25*X^25 + 72*X^24 + 39*X^23 + 16*X^22 + 80*X^21 + 14*X^20 + 35*X^19 + 43*X^18 + 64*X^17 + 27*X^16 + 70*X^15 + 70*X^14 + 40*X^13 + 16*X^12 + 39*X^11 + 79*X^10 + 5*X^9 + 21*X^8 + 51*X^7 + 2*X^6 + 84*X^5 + 96*X^4 + 84*X^3 + 41*X^2 + 14*X + 4]
sage: S = Matrix.diagonal(s)
sage: U = Matrix.random(pR,4,4,"unimodular")
sage: U = U*Matrix.random(pR,4,4,"unimodular")
sage: U = U.T*Matrix.random(pR,4,4,"unimodular").T
sage: V = Matrix.random(pR,4,4,"unimodular").T
sage: V = V*Matrix.random(pR,4,4,"unimodular")
sage: V = V*Matrix.random(pR,4,4,"unimodular")*V
sage: M = U*S*V
sage: SS = M.smith_form()
sage: S == S[0]
False
sage: S == SS[0]
False
sage: s[0]/57 == SS[0,0]
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-18-4d47cbf134aa> in <module>()
----> 1 s[Integer(0)]/Integer(57) == SS[Integer(0),Integer(0)]

TypeError: tuple indices must be integers, not tuple
sage: s[0]/57 == SS[0][0,0]
True
sage: s[3]/47 == SS[0][1,1]
False
sage: s[3]/47 == SS[0][3,3]
False
sage: SS[0][3,3]
86*X^29 + 85*X^28 + 3*X^27 + 37*X^26 + 54*X^25 + 43*X^24 + 92*X^23 + 85*X^22 + 37*X^21 + 38*X^20 + 95*X^19 + 89*X^18 + 49*X^17 + 4*X^16 + 93*X^15 + 93*X^14 + 67*X^13 + 85*X^12 + 92*X^11 + 62*X^10 + 69*X^9 + 57*X^8 + 83*X^7 + 47*X^6 + 34*X^5 + 25*X^4 + 34*X^3 + 42*X^2 + 38*X + 94
sage: s[3]/47 == SS[0][3,3]/86
True
sage: cd ~/repository/polmat/sage/
/home/vneiger/repository/polmat/sage
sage: %runfile init.sage
sage: K = minimal_kernel_basis( M[2:,:] )[0]
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-26-151642b499c1> in <module>()
----> 1 K = minimal_kernel_basis( M[Integer(2):,:] )[Integer(0)]

TypeError: minimal_kernel_basis() takes at least 2 arguments (1 given)
sage: K = minimal_kernel_basis( M[:,:2], [0]*4 )[0]
call with  4 2
degrees -\>
[89 91]
[91 93]
[93 95]
[95 97]
shift -->  [98, 98, 98, 98]
The order seems to be a single integer (call it 'd'), I was expecting a list...
... order converted to the list [d,...,d]
call with  2 1
degrees -\>
[44]
[44]
shift -->  [47, 48]
call with  1 1
degrees -\>
[8]
shift -->  [88]
sage: degree_matrix(K)

[50 50 50 48]
[51 51 50 48]
sage: degree_matrix(M)

[ 89  91  93  95]
[ 91  93  95  97]
[ 93  95  97  99]
[ 95  97  99 101]
sage: K2 = minimal_kernel_basis( M[:,2:], [0]*4 )[0]
call with  4 2
degrees -\>
[ 93  95]
[ 95  97]
[ 97  99]
[ 99 101]
shift -->  [102, 102, 102, 102]
The order seems to be a single integer (call it 'd'), I was expecting a list...
... order converted to the list [d,...,d]
call with  2 1
degrees -\>
[46]
[46]
shift -->  [49, 50]
call with  1 1
degrees -\>
[8]
shift -->  [92]
sage: degree_matrix(K2)

[52 52 52 50]
[53 53 52 50]
sage: KK1 = Matrix.block( [[K],[K2]] )
sage: M2 = KK1 * M
sage: degree_matrix(M2)

[-1 -1 52 54]
[-1 -1 53 55]
[52 52 -1 -1]
[53 53 -1 -1]
sage: KK1 = Matrix.block( [[K2],[K]] )
sage: degree_matrix(M2)

[-1 -1 52 54]
[-1 -1 53 55]
[52 52 -1 -1]
[53 53 -1 -1]
sage: M2 = KK1 * M
sage: degree_matrix(M2)

[52 52 -1 -1]
[53 53 -1 -1]
[-1 -1 52 54]
[-1 -1 53 55]
sage: prod(s).degree()
52
sage: M.det().degree()
52
sage: M2.det().degree()
80
sage: M21 = M2[:2,:2]
sage: M22 = M2[2:,2:]
sage: K211 = minimal_kernel_basis( M21[:,0], [0]*2 )[0]
call with  2 1
degrees -\>
[52]
[53]
shift -->  [54, 54]
sage: K212 = minimal_kernel_basis( M21[:,1], [0]*2 )[0]
call with  2 1
degrees -\>
[52]
[53]
shift -->  [54, 54]
sage: K221 = minimal_kernel_basis( M22[:,0], [0]*2 )[0]
call with  2 1
degrees -\>
[52]
[53]
shift -->  [54, 54]
sage: K222 = minimal_kernel_basis( M22[:,1], [0]*2 )[0]
call with  2 1
degrees -\>
[54]
[55]
shift -->  [56, 56]
sage: KK21 = Matrix.block( [[K211],[K212]] )
sage: KK22 = Matrix.block( [[K221],[K222]] )
sage: M3 = Matrix.block_diagonal( [KK21,KK22] ) * M2
sage: degree_matrix(M3)

[-1 29 -1 -1]
[29 -1 -1 -1]
[-1 -1 -1 29]
[-1 -1 29 -1]
sage: KK21 = Matrix.block( [[K212],[K211]] )
sage: KK22 = Matrix.block( [[K222],[K221]] )
sage: M3 = Matrix.block_diagonal( [KK21,KK22] ) * M2
sage: degree_matrix(M3)

[29 -1 -1 -1]
[-1 29 -1 -1]
[-1 -1 29 -1]
[-1 -1 -1 29]

