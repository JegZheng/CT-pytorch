import torch
import numpy as np

from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_s_curve

def load_data(name='swiss_roll', n_samples=500):
    N=n_samples
    if name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N, noise=0.05)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N, noise=0.02)[0]
        temp/=abs(temp).max()
    elif name == '2gaussians':
        scale = 2.
        centers = [
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '8gaussians':
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=N, noise=0.05)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    elif name == 's_curve':
        temp = make_s_curve(n_samples=500, noise=0.02)[0]
        temp = np.stack([temp[:,0], temp[:,2] ],axis=1)
    else:
        raise Exception("Dataset not found: name must be 'swiss_roll', 'half_moons', 'circle', 's_curve', '8gaussians' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X
