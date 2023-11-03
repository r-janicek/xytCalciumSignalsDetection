
import os
import os.path
from collections import defaultdict
from functools import reduce

import numpy as np

__all__ = ["Summary", "save_npz", "load_npz"]

class Summary:
    
    def __init__(self):
        
        self.content = defaultdict(dict)
    
    def add_scalar(self, tag, value, index):
        self.register(tag, value, index)
    
    def add_scalars(self, tag, values, index):
        for k, v in values.items():
            self.register("{}/{}".format(tag, k), v, index)
    
    def add_image(self, tag, value, index):
        self.register(tag, value, index)
    
    def register(self, tag, value, index):
        self.content[tag][index] = value
    
    def keys(self):
        return self.content.keys()
    
    def get(self, tag, raise_key_error=True):
        
        if tag not in self.content:
            if raise_key_error:
                raise KeyError(tag)
            else:
                return np.array([]), np.array([])
        
        data = self.content[tag]
        
        indices = []
        values = []
        for index in sorted(data):
            indices.append(index)
            values.append(data[index])
        
        return np.asarray(indices), np.asarray(values)
    
    def get_many(self, tags):
        
        dicts = [self.content[tag] for tag in tags]
        indices = [list(d.keys()) for d in dicts]
        indices = reduce(np.intersect1d, indices)
        indices = sorted(indices)
        
        results = tuple([] for tag in tags)
        for index in indices:
            for d, res in zip(dicts, results):
                res.append(d[index])
        
        return indices, results


def save_npz(summary, filename):
    
    res = {}
    for tag in summary.keys():
        indices, values = summary.get(tag)
        res["{}/indices".format(tag)] = indices
        res["{}/values".format(tag)] = values
    
    np.savez(filename, **res)


def load_npz(summary, filename=None):
    
    if filename is None:
        filename = summary
        summary = Summary()
    
    data = np.load(filename)
    
    # Remove the last part from the keys of data
    # (ie, remove "/indices", "/values")
    tags = set(map(lambda x: "/".join(x.split("/")[:-1]), data.keys()))
    
    content = defaultdict(dict)
    for tag in tags:
        indices = data["{}/indices".format(tag)]
        values = data["{}/values".format(tag)]
        
        for index, value in zip(indices, values):
            content[tag][index] = value
    
    summary.content = content
    return summary


# def save_h5(summary, filename):
    
#     f = h5py.File(filename, "w")
#     for tag in summary.content.keys():
#         grp = f.create_group(tag)
        
#         indices, values = summary.get(tag)
        
#         grp.create_dataset("indices", data=indices)
#         grp.create_dataset("values", data=values)
    
#     f.close()

# def load_h5(summary, filename=None):
    
#     if filename is None:
#         filename = summary
#         summary = Summary()
    
#     f = h5py.File(filename, "r")
    
#     content = defaultdict(dict)
    
#     for tag in f.keys():
#         grp = f[tag]
#         indices = grp["indices"][:]
#         values = grp["values"][:]
        
#         for index, value in zip(indices, values):
#             content[tag][index] = value
    
#     f.close()
#     summary.content = content
    
#     return summary
