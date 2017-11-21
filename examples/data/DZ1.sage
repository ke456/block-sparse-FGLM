def eval(x1, x2, x3, x4):
        return [x1^4 + 65536*x2*x3*x4, 65536*x1*x3*x4 + x2^4, 65536*x1*x2*x4 + x3^4, 65536*x1*x2*x3 + x4^4]

