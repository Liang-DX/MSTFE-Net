import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os
import resampy

def load_data(dataset_path, data_file):
    data_path = os.path.join(dataset_path, data_file + '_data.npy')
    label_path = os.path.join(dataset_path, data_file + '_label.npy')
    data = np.load(data_path)
    label = np.load(label_path).squeeze()-1
    print(data_file, 'load success')
    data, label = shuffle_data(data, label)
    return data, label


def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label