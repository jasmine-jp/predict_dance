ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
node = 100000
size = 100
channels, kernel, stride, pool = 10, 5, 5, 2
conv_size = int((size-kernel)/stride/pool)+1
arr_size = int(node / (conv_size**2) / channels)