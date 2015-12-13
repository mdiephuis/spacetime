#!/usr/bin/env python

from __future__ import print_function

import spacetime
import numpy as np
import scipy

EPS = np.finfo( float ).eps

def load( n ):
    C = np.random.uniform( .8, 1, size=(3*n,3*n) )
    for i in range( 1, 4 ):
        C[(i-1)*n:i*n, (i-1)*n:i*n] = np.random.uniform( 0,.2, size=(n,n) )

    for i in range(3*n): C[i,i] = 0
    C = C + C.T
    C /= C.sum()
    C = np.maximum( C, EPS )
    return C

def visualize( z ):

    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    from matplotlib.transforms import Bbox

    fig = plt.figure( figsize=[10,8] )
    fig.suptitle( '' )
    ax = fig.add_subplot( 111 )

    ax.scatter( z[:,0], z[:,1], s=50,
                linewidths=1 )
                #c=z,
                #cmap='RdYlGn',
                #vmin=-np.abs( z ).max(),
                #vmax= np.abs( z ).max() )
    plt.show()

if __name__ == '__main__':
    P = load( 3 )

    spacetime.distribution = 'student'
    spacetime.min_epochs = 500
    spacetime.lrate_s =  5
    spacetime.lrate_t = .01

    #dim = 2
    spacetime_Y,spacetime_Z,E1= spacetime.st_snep( P, 0, 2, repeat=1 )
    #spacetime_Y,spacetime_Z,E2= spacetime.st_snep( P, dim, 0, repeat=10 )
    #print( E1, E2 )

    visualize( spacetime_Z )
    print( spacetime_Z )

