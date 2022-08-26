import cv2, os, pickle, numpy as np
from tqdm import tqdm
from common import size, arr_size, ansmap, lenA

def read(name, terdir):
    name = name.split('.')[0]
    pkl = 'out/src/{}.pkl'.format(name)
    video = '{}/{}.mp4'.format(terdir, name)

    cap = cv2.VideoCapture(video)
    arr = np.array([])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_ma = int((h-w)/2) if w < h else 0
    w_ma = int((w-h)/2) if w > h else 0

    if not os.path.isfile(pkl):
        print('dumping '+video)
        with open(pkl, 'wb') as f:
            for _ in tqdm(range(frame_count)):
                _, frame = cap.read()
                frame = cv2.resize(frame[h_ma:h-h_ma, w_ma:w-w_ma], dsize=(size, size))
                frame = np.array([cv2.split(frame)])
                arr = frame if arr.size == 0 else np.append(arr, frame, axis=0)
            pickle.dump(arr, f)

    print('loading '+pkl)
    with open(pkl, 'rb') as f:
        return pickle.load(f)

def all_read(name):
    if input('update data [y/n]: ') == 'y':
        data, teachs, plot = np.array([]), np.array([]), np.array([])
        other = [0 for _ in range(lenA-1)]+[1]
        for s in os.listdir(name):
            arr = read(s, name)
            data = arr if data.size == 0 else np.append(data, arr, axis=0)
            teach = np.array([ansmap.get(s.split('_')[-1].replace('.mp4', ''), other) for _ in range(len(arr)-arr_size)])
            teachs = teach if teachs.size == 0 else np.append(teachs, teach, axis=0)
            plot = np.append(plot, len(data)-arr_size)
        with open('out/model/{}_read.pkl'.format(name), 'wb') as f:
            print('dumping '+'out/model/{}_read.pkl'.format(name))
            pickle.dump((data, teachs, plot), f)
        return data, teachs, plot
    else:
        with open('out/model/{}_read.pkl'.format(name), 'rb') as f:
            print('loading '+'out/model/{}_read.pkl'.format(name))
            return pickle.load(f)

def test_read():
    if input('read video data [y/n]: ') == 'y':
        with open('out/model/video_read.pkl', 'rb') as f:
            print('loading '+'out/model/video_read.pkl')
            return pickle.load(f)
    else:
        return all_read('test')