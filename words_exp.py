#!/usr/bin/env python

from __future__ import print_function

from nips_exp import find_renderer, overlap_ratio
import spacetime
import scipy.io
import numpy as np
import sys, os
EPS = np.finfo( float ).eps

def __rebuild( data_file, bin_file, dtype ):
    print( 'rebuilding P matrix' )
    raw = scipy.io.loadmat( data_file )
    words = raw['words']
    P     = raw['P'].astype( dtype )
    # note this P is NOT symmetric

    P /= P.sum(1)[:,None]
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, EPS )

    print( "saving to '%s'" % bin_file )
    np.savez( bin_file, P=P, words=words )

def load_words( number=5000, dtype=np.float32 ):
    if not ( number in (1000,5000) ):
        raise RuntimeError( 'number=%d' % number )

    data_file = 'data/association%d.mat' % number
    bin_file  = os.path.splitext( data_file )[0] + '.npz'

    if not os.access( data_file, os.R_OK ):
        raise RuntimeError( "'%s' missing" % data_file )

    if not os.access( bin_file, os.R_OK ):
        __rebuild( data_file, bin_file, dtype )

    print( "loading from '%s'" % bin_file )
    _tmp = np.load( bin_file )
    return _tmp['P'], _tmp['words']

def __embed( P, result_file, repeat ):
    '''
    (optionally) compute the embeding and save it to disk
    then load the embedding from disk
    '''
    if not os.access( result_file, os.R_OK ):
        spacetime.conv_threshold = 1e-9
        spacetime.min_epochs     = 1000
        spacetime.lrate_s = 500
        spacetime.lrate_t = 1

        Y, Z, E = spacetime.st_snep( P, 2, 1, repeat=repeat )
        np.savez( result_file, Y=Y, Z=Z, E=E )

    print( 'loading results from "%s"' % result_file )
    tmp = np.load( result_file )
    return tmp['Y'], tmp['Z'], tmp['E']

def __visualize( Y, Z, words, fig_file ):

    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc( 'pdf',  fonttype=42 )
    rc( 'ps',   fonttype=42 )

    # visualize
    fig = plt.figure( figsize=(8,8), dpi=600, frameon=True )
    ax = plt.Axes( fig, [0, 0, 1, 1] )
    ax.set_aspect( 'equal' )
    ax.set_xticks( [ -150, -100, -50, 0, 50, 100, 150] )
    ax.set_yticks( [ -150, -100, -50, 0, 50, 100, 150] )
    fig.add_axes( ax )

    to_show = []
    for i in range( Y.shape[0] ):
        if len( to_show ) > 0:
            dist = np.sqrt( (( Y[to_show] - Y[i] )**2).sum(1) )
            if dist.min() < 6: continue
        to_show.append( i )
    print( 'showing %d words' % len(to_show) )

    cNorm     = colors.Normalize( vmin=-.8, vmax=.8 )
    scalarMap = cmx.ScalarMappable( norm=cNorm, cmap='RdYlBu_r' )

    scale_Z = np.sqrt( (Z**2).sum(1) )
    font_s = (scale_Z      -scale_Z.min()) / \
             (scale_Z.max()-scale_Z.min()) * 11 + 2

    offset = .01 * ( plt.xlim()[1]-plt.xlim()[0] )
    past_bb = []
    for i in to_show:
        tt = ax.text( Y[i,0], Y[i,1], words[i][0][0],
                      size=font_s[i],
                      rotation=0,
                      color=scalarMap.to_rgba( Z[i,0] ),
                      alpha = .9,
                      va='center', ha='center' )
        transf = ax.transData.inverted()
        bb = tt.get_window_extent( renderer = find_renderer( fig ) ).transformed( transf ).get_points()

        if overlap_ratio( past_bb, bb ) > .02:
            best_o      = np.inf
            best_adjust = None
            search_range = np.vstack( [ np.linspace( 0,  5*offset, 20 ),
                                        np.linspace( 0, -5*offset, 20 ) ]
                                    ).flatten('F')
            for x_adjust in search_range:
                for y_adjust in search_range:
                    oratio = overlap_ratio( past_bb, bb + np.array([x_adjust, y_adjust]) )
                    if oratio < best_o * .95:
                        best_o = oratio
                        best_adjust = np.array( [x_adjust,y_adjust] )
            bb += best_adjust
            tt.set_x( .5*(bb[0,0]+bb[1,0]) )
            tt.set_y( .5*(bb[0,1]+bb[1,1]) )
        past_bb.append( bb )

    x_min = Y[to_show,0].min()
    x_max = Y[to_show,0].max()
    x_gap = 0 * (x_max-x_min)
    y_min = Y[to_show,1].min()
    y_max = Y[to_show,1].max()
    y_gap = 0 * (y_max-y_min)
    plt.xlim( x_min-x_gap, x_max+x_gap )
    plt.ylim( y_min-y_gap, y_max+y_gap )

    # histogram of Z
    ax_inset = plt.axes( (0.03, 0.05, 0.2, 0.2), frameon=False )
    counts, bins, patches = ax_inset.hist( Z, 9, fc='0.5', ec='gray' )
    ax_inset.xaxis.set_ticks_position( "none" )
    ax_inset.yaxis.set_ticks_position( "right" )
    plt.xticks( [-1.5, 0, 1.5], size=8 )
    plt.yticks( [500, 1000],    size=8 )
    for _bin, _patch in zip( bins, patches ):
        _patch.set_facecolor( scalarMap.to_rgba( _bin ) )
    ax_inset.set_title( 'histogram of time coordinates', size=9 )

    # colorbar
    scalarMap._A = []
    cax  = fig.add_axes( [1.01, 0.01, 0.04, 0.98] )
    cbar = fig.colorbar( scalarMap, ticks=[-.8, -.4, 0, .4, .8], cax=cax )
    cax.text( .5, .5, '---time-->', size=14, rotation=90, va='center', ha='center' )
    cbar.ax.yaxis.set_ticks_position( 'right' )
    cbar.ax.set_yticklabels( ['<-0.8', '-0.4', '0', '0.4', '>0.8'] )

    print( 'printing visualization to "%s"' % fig_file )
    fig.savefig( fig_file,
                 transparent=True,
                 bbox_inches='tight',
                 edge_color='white',
                 pad_inches=0 )

if __name__ == '__main__':
    P, words = load_words( 5000 )
    print( '%d words' % P.shape[0] )

    if len( sys.argv ) > 1:
        result_file = 'results/words_result_%s.npz' % sys.argv[1]
    else:
        result_file = 'results/words_result.npz'

    Y, Z, E = __embed( P, result_file, 1 )

    # show some ranking results
    print( 'E=%.3f' % E )

    scale_Z = np.sqrt( (Z**2).sum(1) )
    scale_Y = np.sqrt( (Y**2).sum(1) )

    rank = np.argsort( scale_Z )[::-1]
    words_to_show = 25
    print( 'top %d words by z:' % words_to_show )
    for i in rank[:words_to_show]:
        print( '%20s   z=%.2f' % ( words[i][0][0], Z[i,0] ) )
    print( 'bottom %d words:' % words_to_show )
    for i in rank[-words_to_show:]:
        print( '%20s   z=%.2f' % ( words[i][0][0], Z[i,0] ) )

    fig_file = os.path.splitext( result_file )[0] + '.pdf'
    __visualize( Y, Z, words, fig_file )

