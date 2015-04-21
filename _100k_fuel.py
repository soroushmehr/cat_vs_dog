from collections import OrderedDict


from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes

import numpy as np


@do_not_pickle_attributes('indexables')
class _100k(IndexableDataset):
    provides_sources = ('features', 'target')

    def __init__(self, which_set, **kwargs):
        self.data = np.load('/data/lisatmp3/mehris/catdog_patches/vae100k16x16-white-f32.npy')
        #self.data = np.load('/data/lisatmp3/mehris/catdog_patches/toy-white-f32-PCA.npy')
        print "shape: ", self.data.shape
        trainInd = self.data.shape[0]*0.8
        if which_set is 'train':
            self.data = self.data[:trainInd]
            print 'train', self.data.shape
        elif which_set is 'valid':
            self.data = self.data[trainInd:]
            print 'valid', self.data.shape
        else:
            print "NONE"

        super(_100k, self).__init__(OrderedDict(zip(self.provides_sources,
                                   self._load_vae_100k())), **kwargs)

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_vae_100k())
                           if source in self.sources]

    def _load_vae_100k(self):
        return self.data, np.zeros(self.data.shape[0])
