#!/usr/bin/env python

from __future__ import print_function

import spacetime
import numpy as np
import scipy

EPS = np.finfo( float ).eps

def __class( n ):
    C = np.zeros( [n+1,n+1] )
    for i in range( n ):
        C[i,(i-2)%n] = 1
        C[i,(i-1)%n] = 1
        C[i,(i+1)%n] = 1
        C[i,(i+2)%n] = 1
        C[i,n] = C[n,i] = 1
    return C

def load_school( n=20 ):
    C = scipy.linalg.block_diag( __class(n), __class(n) )
    C[n,2*n+1] = C[2*n+1,n] = 1
    C /= C.sum()
    C = np.maximum( C, EPS )
    return C

def visualize( y, z ):

    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    from matplotlib.transforms import Bbox

    fig = plt.figure( figsize=[10,8] )
    fig.suptitle( '' )
    ax = fig.add_subplot( 111 )

    if z.size > 0:
        ax.scatter( y[:,0], y[:,1], s=50,
                    linewidths=1,
                    c=z,
                    cmap='RdYlGn',
                    vmin=-np.abs( z ).max(),
                    vmax= np.abs( z ).max() )
    else:
        z = y[:,2]
        ax.scatter( y[:,0], y[:,1], s=50,
                    linewidths=1,
                    c=z,
                    cmap='RdYlGn',
                    vmin=-np.abs( z ).max(),
                    vmax= np.abs( z ).max() )

    plt.show()

if __name__ == '__main__':
    P = load_school()
    spacetime.distribution = 'student'
    spacetime.lrate_s =  1
    spacetime.lrate_t = .01

    dim = 3
    #spacetime_Y,spacetime_Z,E1= spacetime.st_snep( P, dim-1, 1, repeat=1 )
    spacetime_Y,spacetime_Z,E2= spacetime.st_snep( P, dim, 0, repeat=3 )
    #print( E1, E2 )

    visualize( spacetime_Y, spacetime_Z )
    #print( spacetime_Z )

