import os
import pickle

def writeToPickle(obj, path, iter=0):
    base_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    f = open(path, "wb")
    pickle.dump([obj, iter], f)
