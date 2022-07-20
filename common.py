ansmap = {
    'elegant': [1, 0, 0],
    'dance': [0, 1, 0]
}
node = 60000
size = 60
channels, kernel, stride, pool = 6, 3, 3, 2
conv_size = int((size-kernel)/stride/pool)+1
arr_size = int(node / (conv_size**2) / channels)