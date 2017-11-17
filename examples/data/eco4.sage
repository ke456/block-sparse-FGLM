def eval(x1, x2, x3, x4):
        return [x1*x2*x3 + x1*x3*x4 + x1*x4 + 65536, x1*x2*x4 + x1*x3 + 65535, x1*x2 + 65534, x2 + x3 + x4 + 1]

