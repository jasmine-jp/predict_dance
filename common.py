from sympy import factorint

ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
size, arr_size, pool = 60, 90, 2
third, second = factorint(int(size/pool/pool)).keys()
diff = int(size/second/third/pool/pool)
channel = int(size/second/pool/diff)
batch, hidden, rang = 10, int(arr_size/2), 3