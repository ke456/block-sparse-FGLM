def eval(x1, x2, x3, x4, x5):
        return [65535*x1 + 65536*x2 + 65533*x3 + 65534*x4 + 65535*x5, 3*x1 + x2 + 65533*x4 + 4*x5, 4*x1 + 65535*x2 + 5*x3 + 3*x4 + 65532*x5, 65536*x1^5 + 5*x1*x2*x3*x4*x5 + 65536*x2^5 + 65536*x3^5 + 65536*x4^5 + 
65536*x5^5, x1^3*x2*x5 + x1*x2^3*x3 + x1*x4*x5^3 + x2*x3^3*x4 + x3*x4^3*x5, x1^2*x2*x3^2 + x1^2*x4^2*x5 + x1*x2^2*x5^2 + x2^2*x3*x4^2 + x3^2*x4*x5^2, x1^2*x2^3*x4*x5 + x1^2*x2*x3*x5^3 + 65536*x1*x2^2*x3^4 + 
65535*x1*x2*x3^2*x4^2*x5 + 65536*x1*x4^4*x5^2 + x2^5*x3*x4 + x3*x4*x5^5, x1^4*x2^2*x3 + 2*x1^2*x2*x3*x4*x5^2 + 65536*x1*x2^5*x5 + 65536*x1*x2*x3^2*x4^3 +
65536*x1*x4^5*x5 + 65536*x2^3*x3^2*x4*x5 + x3*x4^2*x5^4, 65536*x1^6*x2 + 65536*x1^4*x4*x5^2 + 3*x1^2*x2^2*x3*x4*x5 + 65536*x1*x2^6 + 
65536*x1*x2*x4^5 + x1*x3^3*x4^2*x5 + 65536*x2^4*x3^2*x4 + x2*x3*x4^2*x5^3, 65536*x1^5*x2*x3 + 65536*x1^3*x3*x4*x5^2 + x1^2*x2^4*x5 + 2*x1*x2^2*x3^2*x4*x5 +
65536*x1*x2*x4^3*x5^2 + 65536*x2*x3*x4^5 + x3^4*x4^2*x5, 65536*x1^3*x2^2*x3*x4 + x1^2*x2*x5^4 + 65536*x1*x2^2*x3^3*x5 + 
65534*x1*x2*x3*x4^2*x5^2 + x2^5*x4*x5 + x2*x3^2*x4^4 + x4^6*x5 + x4*x5^6]
