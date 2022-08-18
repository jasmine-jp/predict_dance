from sympy import factorint

ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
ans = len(ansmap)+1
size, arr_size, pool = 60, 90, 2
second, third = list(factorint(int(size/pool/pool)).keys())[-1:-3:-1]
diff = int(size/second/third/pool/pool)
channel = int(size/second/pool)
batch, hidden = 10, arr_size