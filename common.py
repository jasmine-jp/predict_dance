from sympy import factorint

ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0],
    'others': [0, 0, 1]
}
size, arr_size, pool = 60, 64, 2
third, second = factorint(int(size/pool/pool)).keys()
diff = int(size/second/third/pool/pool)
channel = int(size/second/pool/diff)