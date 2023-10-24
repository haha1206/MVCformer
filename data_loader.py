import numpy as np
import scipy.io as sio

def load_mat(path):

    data = sio.loadmat(path)

    X = np.squeeze(data['distance_matrices'])

    Y = np.squeeze(data['gt'])

    return X,Y









