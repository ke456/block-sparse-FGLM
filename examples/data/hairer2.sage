def eval(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
        return [65533*x1 + x2 + 65535*x4 + 3*x5 + 65532*x6 + x7 + 4*x8 + x9 + 65535*x10 + x11 + 
65533*x12 + 65534*x13, 3*x1 + 65532*x2 + 65534*x3 + 65535*x4 + 5*x5 + 65536*x6 + 65533*x7 + 65532*x8 + 
x9 + 65536*x10 + 65534*x11 + 4*x12 + 5*x13, x4 + x5 + x6 + x7 + 65536, 2*x1*x6 + 2*x2*x5 + 2*x3*x4 + 65536, 3*x1^2*x6 + 3*x2^2*x5 + 3*x3^2*x4 + 65536, 6*x1*x4*x12 + 6*x1*x5*x9 + 6*x2*x4*x13 + 65536, 4*x1^3*x6 + 4*x2^3*x5 + 4*x3^3*x4 + 65536, 8*x1*x2*x5*x9 + 8*x1*x3*x4*x12 + 8*x2*x3*x4*x13 + 65536, 2*x1^2*x4*x12 + 2*x1^2*x5*x9 + 2*x2^2*x4*x13 + 65536, 24*x1*x4*x9*x13 + 65536, x1 + 65536*x8, x2 + 65536*x9 + 65536*x10, x3 + 65536*x11 + 65536*x12 + 65536*x13]

