#!/usr/bin/env python
import numpy as np

from scipy.stats import norm, chi2

def sample_checkerboard(nsamp,board_size=3):
    s = np.random.uniform(size=(2,0))
    for i in range(board_size):
        off_x = i%2
        for j in range(board_size-off_x):
            s = np.hstack([s,np.random.uniform(size=(2,nsamp))+np.array([i,1*j+off_x])[:,np.newaxis]])
    return s

def gaussian_peaks(nsamp,npeaks):
    samp = np.random.normal(size=(2,0))
    for i in range(npeaks):
        for j in range(npeaks):
            samp = np.hstack( [samp, np.random.normal(size=(2,nsamp)) + np.arraY9[10*i,10*j])[:,np.newaxis] ])
    return samp

def local_mapping(nSamp,nLat,f):
    samp = np.random.normal(size=(nLat,nSamp))
    return f(samp)

if __name__=="__main__":
    pass
