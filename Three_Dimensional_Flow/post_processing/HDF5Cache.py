import hashlib
import h5py
import numpy as np
import copy

class HDF5Cache(object):
    def __init__(
            self,
            filename = 'HDF5Cache_test.h5',
            read_only = True):
        self.read_only = read_only
        if self.read_only:
            self.file = h5py.File(filename, 'r')
        else:
            self.file = h5py.File(filename, 'a')
        self.group = self.file
        self.key = ''
        return None
    def reset_key(
            self,
            key_object):
        try:
            key_object_iterator = iter(key_object)
            self.key = hashlib.md5(bytes(
                '-'.join([str(aa) for aa in key_object_iterator]), encoding = 'utf-8')).hexdigest()
        except TypeError:
            self.key = hashlib.md5(bytes(str(key_object), encoding = 'utf-8')).hexdigest()
        if self.key in self.file.keys():
            self.group = self.file[self.key]
        else:
            if self.read_only:
                print('Error: trying to create group in read-only cache file.')
            self.group = self.file.create_group(self.key)
        self.key_object = copy.deepcopy(key_object)
        return None
    def __getitem__(
            self,
            key):
        return self.group[key]
    def __delitem__(
            self,
            key):
        del self.group[key]
        return None
    def __setitem__(
            self,
            key,
            value):
        self.group[key] = value
        return None
    def keys(self):
        return self.group.keys()

def main():
    return None

if __name__ == '__main__':
    main()

