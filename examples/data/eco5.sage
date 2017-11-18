def eval(x1, x2, x3, x4, x5):
        return [x1*x2*x3 + x1*x3*x4 + x1*x4*x5 + x1*x5 + 65536, x1*x2*x4 + x1*x3*x5 + x1*x4 + 65535, x1*x2*x5 + x1*x3 + 65534, x1*x2 + 65533, x2 + x3 + x4 + x5 + 1]

