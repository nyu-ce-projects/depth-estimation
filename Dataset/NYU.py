import os

import h5py
import numpy as np

class NYU(object):

    def __init__(self):
        datasetPath = os.path.join(os.getcwd(), "data", "NYU", "nyu_depth_v2_labeled.mat")
        self.dataset = h5py.File(datasetPath, "r")
        print(self.dataset["images"][0])

if __name__ == "__main__":
    nyu = NYU()
