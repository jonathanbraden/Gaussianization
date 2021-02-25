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

def gaussian_peak_grid(nSamp,nDim,nPeaks):
    samp = np.random.normal(size=(nDim,0))
    for i in range(nPeaks):
        for j in range(nPeaks):
            samp = np.hstack( [samp, np.random.normal(size=(nDim,nSamp)) + np.array([10*i,10*j])[:,np.newaxis] ])
    return samp

def gaussian_peaks_random(nSamp,nDim,nPeaks):
    samp = np.random.normal(size=(nDim,0))
    # Generate random peak locations
    loc = np.random.uniform( size=(nDim,nPeaks) )

def gaussian_peaks(nsamp,npeaks):
    samp = np.random.normal(size=(2,0))
    for i in range(npeaks):
        for j in range(npeaks):
            samp = np.hstack( [samp, np.random.normal(size=(2,nsamp)) + np.array([10*i,10*j])[:,np.newaxis] ])
    return samp

def local_mapping(nSamp,nLat,f):
    samp = np.random.normal(size=(nLat,nSamp))
    return f(samp)

def ring_pdf(nsamp):
    return

if __name__=="__main__":
    pass
