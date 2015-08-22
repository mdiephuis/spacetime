#!/usr/bin/env python

from __future__ import print_function
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

import spacetime
import scipy.io
import numpy as np
import sys, os
EPS = np.finfo( float ).eps

def load_words( number=1000 ):
    data_file = 'data/association%d.mat' % number
    bin_file  = os.path.splitext( data_file )[0] + '.npz'
    if not os.access( bin_file, os.R_OK ):
        print( 'rebuilding P matrix' )
        raw = scipy.io.loadmat( data_file )
        words = raw['words']
        P     = raw['P'].astype( np.float32 )

        P /= P.sum(1)[:,np.newaxis]
        P = P + P.T
        P /= P.sum()
        P = np.maximum( P, EPS )
        np.savez( bin_file, P=P, words=words )

    _tmp = np.load( bin_file )
    return _tmp['P'], _tmp['words']

if __name__ == '__main__':
    P, words = load_words()
    print( '%d words' % P.shape[0] )

    result_file = 'spacetime_words_result.npz'
    if os.access( result_file, os.R_OK ):
        print( 'using existing results in %s' % result_file )
        tmp = np.load( result_file )
        Y = tmp['Y']
        Z = tmp['Z']
        E = tmp['E']
        print( 'E=%.3f' % E )
    else:
        spacetime.conv_threshold = 1e-9
        Y, Z, E = spacetime.st_snep( P, 2, 1, repeat=10 )
        np.savez( result_file, Y=Y, Z=Z, E=E )

    scale_Z = np.sqrt( (Z**2).sum(1) )
    scale_Y = np.sqrt( (Y**2).sum(1) )
    #plt.hist( scale_z ); plt.show()

    rank = np.argsort( scale_Z )[::-1]
    print( 'top 20 words by z:' )
    for i in rank[:20]:
        print( '%20s   z=%.2f' % ( words[i][0][0], Z[i,0] ) )
    print( 'bottom 20 words:' )
    for i in rank[-20:]:
        print( '%20s   z=%.2f' % ( words[i][0][0], Z[i,0] ) )

    outf = plt.figure( figsize=(10,8), frameon=True )

    idx = []
    scatter = np.zeros( [0,2] )
    for i in range( P.shape[0] ):
        if scatter.shape[0] > 0:
            dist = np.sqrt( (( scatter - Y[i] )**2).sum(1) )
            if dist.min() < 9: continue
        idx.append( i )
        scatter = np.vstack( [scatter, Y[i]] )
    idx = np.array( idx )

    cNorm     = colors.Normalize( vmin=-scale_Z.max(), vmax=scale_Z.max() )
    scalarMap = cmx.ScalarMappable( norm=cNorm, cmap='jet' )

    ax = outf.add_subplot( 111 )
    ax.scatter( Y[idx,0], Y[idx,1], s=5, c='.5', linewidths=0 )
    font_s = (scale_Z      -scale_Z.min()) / \
             (scale_Z.max()-scale_Z.min()) * 12 + 2      # from 12 to 14
    for i in idx:
        if scale_Z[i] > .9:
            plt.annotate( words[i][0][0],
                    xy = (Y[i,0], Y[i,1]),
                    xytext = (-10, 0),
                    size = font_s[i],
                    textcoords = 'offset points',
                    ha = 'right', va = 'bottom', 
                    color = scalarMap.to_rgba( Z[i,0] ) )
        else:
            plt.annotate( words[i][0][0],
                    xy = (Y[i,0], Y[i,1]),
                    xytext = (-10, 0),
                    size = font_s[i],
                    textcoords = 'offset points',
                    ha = 'right', va = 'bottom', )

    x_min = Y[idx,0].min()
    x_max = Y[idx,0].max()
    x_gap = .02 * (x_max-x_min)
    y_min = Y[idx,1].min()
    y_max = Y[idx,1].max()
    y_gap = .02 * (y_max-y_min)
    plt.xlim( x_min-x_gap, x_max+x_gap )
    plt.ylim( y_min-y_gap, y_max+y_gap )

    scalarMap.set_array( [-scale_Z.max(), scale_Z.max()] )
    cbar = plt.colorbar( scalarMap )
    cbar.ax.set_aspect( 40 )

    outf.savefig( 'spacetime_words.pdf',
                  transparent=True,
                  bbox_inches='tight',
                  edge_color='white',
                  pad_inches=0 )
    plt.show()

