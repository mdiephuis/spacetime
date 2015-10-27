#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import spacetime
import scipy.io
import sys

def teapot():
    mat = scipy.io.loadmat( '../data/tea.mat' )
    return mat['Input']['X'][0,0].T

def circle( N, noise=0 ):
    x = np.array( [ np.linspace( 0, 2*np.pi, N ) ] ).T
    data = np.hstack( [ np.cos(x), np.sin(x), np.zeros( [N, 98] ) ] )
    data += np.random.randn( data.shape[0], data.shape[1] ) * noise

    return data.astype( np.float32 )

def star( N, noise=0 ):
    n = N/5
    x = np.array( [ np.linspace( 0, 1, n ) ] ).T
    x = np.hstack( [x, np.zeros([n,1]) ] )
    data = []
    for i in range( 5 ):
        a = 2 * np.pi * i / 5 + np.pi / 2
        rot = np.array( [ [np.cos(a), np.sin(a)],
                          [-np.sin(a), np.cos(a)]  ] )
        data.append( np.dot(x, rot) )
    data = np.vstack( data )
    data = np.hstack( ( data, np.zeros( [N, 98] ) ) )
    data += np.random.randn( data.shape[0], data.shape[1] ) * noise
    return data

data = circle( 300, 0.15 )
#plt.scatter( data[:,0], data[:,1] ); plt.show()
#data = teapot()

_f, (ax1,ax2,ax3) = plt.subplots( 3 )
spacetime.distribution = 'gaussian'
spacetime.conv_threshold = 1e-9
spacetime.min_epochs     = 500
Y, Z, E_sne = spacetime.st_sne( data, 3, 0, perplexity=10 )
ax1.scatter( Y[:,0], Y[:,1] )

spacetime.distribution = 'student'
Y, Z, E_tsne = spacetime.st_sne( data, 3, 0, perplexity=10 )
ax2.scatter( Y[:,0], Y[:,1] )

spacetime.lrate_t = .1
Y, Z, E_st = spacetime.st_sne( data, 2, 1, perplexity=10 )
ax3.scatter( Y[:,0], Y[:,1] )

print( 'sne',       E_sne )
print( 'tsne',      E_tsne )
print( 'spacetime', E_st )
plt.show()
