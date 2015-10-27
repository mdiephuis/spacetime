#!/usr/bin/env python

from __future__ import print_function

import dist2, spacetime
import numpy as np
import scipy.io
import os, sys, itertools

EPS    = np.finfo( float ).eps

def __rebuild( data_file, bin_file, num_vols, dtype, binary ):
    '''
    parse the raw data in data_file and
    store the co-authorship matrix in bin_file
    '''
    print( 'rebuilding P matrix' )
    raw = scipy.io.loadmat( data_file )

    if num_vols == 17:
        documents = np.array( raw['docs_authors'].todense(), dtype=np.uint8 )
        authors   = raw['authors_names'][0]
    elif num_vols == 22:
        documents = np.array( raw['documents'].todense(), dtype=np.uint8 )
        authors   = raw['authors'][0]
    else:
        raise RuntimeError( 'num_vols=%d' % num_vols )

    N = authors.shape[0]
    assert( N == documents.shape[1] )
    print( 'originally, %d authors, %d papers in total' % \
           ( N, documents.shape[0] ) )

    # building the co-authorship matrix within authors with 2 papers
    have_two_papers = ( documents.sum(0) >= 2 )
    C = np.zeros( [N, N], dtype=dtype )
    for doc in documents:
        for a1, a2 in itertools.combinations( np.nonzero( doc )[0], 2 ):
            if have_two_papers[a1] and have_two_papers[a2]:
                C[a1, a2] += 1
                C[a2, a1] += 1

    idx = ( C.sum(0) >= 1 )
    print( 'removing %d young authors' % ( C.shape[0] - idx.sum() ) )

    C = C[idx][:,idx]
    if binary:
        C = ( C > 0 ).astype( dtype )
        print( 'binarized with density %.2f%%' % (C.sum()*100./C.size) )
    assert( np.allclose( C, C.T ) )

    authors = authors[idx]
    documents = documents[:, idx]
    print( '%d authors left' % C.shape[0] )
    print( "they co-authored %d papers" % ( documents.sum(1) >= 2 ).sum() )

    # normalize P
    P = C.copy()
    P /= P.sum(0)
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, EPS )

    print( "saving to '%s'" % bin_file )
    np.savez( bin_file, C=C, P=P, authors=authors, no_papers=documents.sum( 0 ) )

def load_nips( num_vols=22, dtype=np.float32, binary=False ):
    '''load the NIPS co-authorship dataset'''

    if not ( num_vols in (17,22) ):
        raise RuntimeError( 'num_vols=%d' % num_vols )

    data_file = 'data/nips_1-%d.mat' % num_vols
    bin_file  = os.path.splitext( data_file )[0] + '.npz'

    if not os.access( data_file, os.R_OK ):
        raise RuntimeError( "'%s' missing" % data_file )

    if not os.access( bin_file, os.R_OK ):
        __rebuild( data_file, bin_file, num_vols, dtype, binary )

    print( 'loading nips data from %s' % bin_file )
    _tmp       = np.load( bin_file )
    return _tmp['C'], _tmp['P'], _tmp['authors'], _tmp['no_papers']

def overlap_ratio( past_bb, bb ):
    o_ratio = 0
    for pp in past_bb:
        x_overlap = max( 0, min(pp[1,0],bb[1,0])-max(pp[0,0],bb[0,0]) )
        y_overlap = max( 0, min(pp[1,1],bb[1,1])-max(pp[0,1],bb[0,1]) )
        o_ratio += (x_overlap * y_overlap )
    area = (bb[1,0]-bb[0,0])*(bb[1,1]-bb[0,1])
    return (o_ratio * 1.0 / area )

def find_renderer( fig ):
    if hasattr(fig.canvas, "get_renderer"):
        renderer = fig.canvas.get_renderer()
    else:
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return( renderer )

def __visualize( Y, Z, authors, no_papers, ofile ):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.lines as lines
    import matplotlib.cm as cmx
    from matplotlib import rc
    rc( 'pdf',  fonttype=42 )
    rc( 'ps',   fonttype=42 )

    fig = plt.figure( figsize=[8,8], frameon=True, dpi=600 )
    ax = plt.Axes( fig, [0., 0., 1., 1.] )
    ax.set_aspect( 'equal' )
    fig.add_axes( ax )

    guys = np.logical_or( ( no_papers >= 10 ),
                          ( np.abs( Z[:,0] ) > 1 ) )
    others = np.logical_not( guys )

    cNorm     = colors.Normalize( vmin=-1, vmax=1 )
    scalarMap = cmx.ScalarMappable( norm=cNorm, cmap='RdYlBu_r' )
    ax.scatter( Y[others,0], Y[others,1],
                s=15,
                c=Z[others,0],
                cmap='RdYlBu_r',
                norm=cNorm,
                alpha=.4,
                edgecolors='none' )

    x_min = Y[guys,0].min()
    x_max = Y[guys,0].max()
    x_gap = .02 * (x_max-x_min)
    y_min = Y[guys,1].min()
    y_max = Y[guys,1].max()
    y_gap = .02 * (y_max-y_min)
    plt.xlim( x_min-x_gap, x_max+x_gap )
    plt.ylim( y_min-y_gap, y_max+y_gap )

    if False:
        scale_plot = .5 * ( (x_max-x_min) + (y_max-y_min) )
        connections = np.transpose( np.nonzero( C > 2 ) )
        violate = 0
        for a1, a2 in connections:
            if np.sqrt( ((y[a1]-y[a2])**2).sum() ) > .3 * scale_plot:
                ax.add_line( lines.Line2D( [y[a1,0], y[a2,0]], [y[a1,1], y[a2,1]],
                    linewidth=1, color='r', alpha=.5 ) )
                violate += 1
        print( violate )

    offset = .01 * ( plt.xlim()[1]-plt.xlim()[0] )
    font_s = np.abs(Z) * 9
    #alpha = np.minimum( np.maximum( (no_papers-10) / 10., 0 ), 1 )
    #alpha = alpha * .6 + .35
    text_positions = Y

    past_bb = []
    for i in np.nonzero( guys )[0]:
        _x = text_positions[i][0]
        _y = text_positions[i][1]
        _a = authors[i][0].split('_')[0]
        tt = ax.text( _x, _y, _a,
                      size=font_s[i],
                      rotation=0,
                      color=scalarMap.to_rgba( Z[i,0] ),
                      alpha = .9,
                      verticalalignment='center',
                      horizontalalignment='center' )
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
                    oratio = overlap_ratio( past_bb, bb + np.array([x_adjust,y_adjust]) )
                    if oratio < best_o * .95:
                        best_o = oratio
                        best_adjust = np.array( [x_adjust,y_adjust] )

            #if best_o > .02 and no_papers[i] < 15 and not (_a in whitelist):
            #    tt.set_alpha( 0 )
            #    print( 'sorry %20s %5.2f (%2d NIPS papers)' % (_a, best_o, no_papers[i] ) )
            #else:
            bb += best_adjust
            tt.set_x( .5*(bb[0,0]+bb[1,0]) )
            tt.set_y( .5*(bb[0,1]+bb[1,1]) )

        past_bb.append( bb )

    # histogram of Z
    ax_inset = plt.axes( (0.77, 0.03, 0.2, 0.2), frameon=False )
    counts, bins, patches = ax_inset.hist( Z, 9, fc='0.5', ec='gray' )
    ax_inset.xaxis.set_ticks_position( "none" )
    ax_inset.yaxis.set_ticks_position( "left" )
    plt.xticks( [-1.5, 0, 1.5], size=8 )
    plt.yticks( [50, 100, 150, 200, 250], size=8 )
    for _bin, _patch in zip( bins, patches ):
        _patch.set_facecolor( scalarMap.to_rgba( _bin ) )
    ax_inset.set_title( 'histogram of time coordinates', size=9 )

    # colorbar
    scalarMap._A = []
    cax  = fig.add_axes( [1.01, 0, 0.04, 1] )
    cbar = fig.colorbar( scalarMap, ticks=[-1, -.5, 0, .5, 1], cax=cax )
    cax.text( .5, .5, '---time-->', size=14, rotation=90, verticalalignment='center', horizontalalignment='center' )
    cbar.ax.yaxis.set_ticks_position( 'right' )
    cbar.ax.set_yticklabels( ['<-1.0', '-0.5', '0', '0.5', '>1.0'] )

    # axes
    ax.tick_params( right='off', top='off' )
    ax.set_xticks( [ -250,   -150,   0,   150,   250] )
    ax.set_yticks( [ -250,   -150,   0,   150,   250] )

    fig.savefig( ofile,
                 bbox_inches='tight',
                 pad_inches=0,
                 transparent=True )

def __embed( P, result_file, methods, repeat ):
    '''
    (optionally) compute the embeding and save to disk
    then load the embedding from disk
    '''
    if not os.access( result_file, os.R_OK ):

        # some good configurations for NIPS22
        spacetime.conv_threshold = 1e-9
        spacetime.min_epochs     = 500
        spacetime.lrate_s        = 500
        spacetime.lrate_t        = 1

        sne_Y = None
        sne_E = 0
        if 'sne' in methods:
            spacetime.distribution = 'gaussian'
            sne_Y,_tmp,sne_E = spacetime.st_snep( P, 3, 0, repeat=repeat )

        tsne_Y = None
        tsne_E = 0
        if 'tsne' in methods:
            spacetime.distribution = 'student'
            tsne_Y,_tmp,tsne_E = spacetime.st_snep( P, 3, 0, repeat=repeat )

        spacetime_Y = spacetime_Z = None
        spacetime_E = 0
        if 'st' in methods:
            spacetime.distribution = 'student'
            spacetime_Y,spacetime_Z,spacetime_E = \
                spacetime.st_snep( P, 2, 1, repeat=repeat )

        np.savez( result_file, 
                  sne_Y=sne_Y,
                  sne_E=sne_E,
                  tsne_Y=tsne_Y,
                  tsne_E=tsne_E,
                  spacetime_Y=spacetime_Y,
                  spacetime_Z=spacetime_Z,
                  spacetime_E=spacetime_E,
                )

    print( 'loading results from %s' % result_file )
    tmp = np.load( result_file )
    return ( tmp['sne_Y'], tmp['sne_E'],
             tmp['tsne_Y'], tmp['tsne_E'], 
             tmp['spacetime_Y'], tmp['spacetime_Z'], tmp['spacetime_E'] )

if __name__ == '__main__':
    REPEAT  = 50
    METHODS = ['st'] #[ 'sne', 'tsne', 'st' ]

    C, P, authors, no_papers = load_nips()
    print( "%d authors" % C.shape[0] )
    big_guys = np.nonzero( no_papers >= 10 )[0]
    print( "%d authors have >=10 NIPS papers" % big_guys.size )

    if len( sys.argv ) > 1:
        result_file = 'results/nips_result_%s.npz' % sys.argv[1]
    else:
        result_file = 'results/nips_result.npz'

    sne_Y, sne_E, tsne_Y, tsne_E, \
        spacetime_Y, spacetime_Z, spacetime_E \
            = __embed( P, result_file, METHODS, REPEAT )

    # for single space time embedding
    scale_Z = np.sqrt( (spacetime_Z**2).sum(1) )
    scale_Y = np.sqrt( (spacetime_Y**2).sum(1) )

    rank = np.argsort( scale_Z )[::-1]
    print( 'top 25 authors by z:' )
    for i in rank[:25]:
        print( '%20s z=%7.3f papers=%2d' % \
               ( authors[i][0], spacetime_Z[i,0], no_papers[i] ) )

    rank_paper = np.argsort( no_papers )[::-1]
    print( 'top 25 authors by #papers:' )
    for i in rank_paper[:25]:
        print( '%20s z=%7.3f papers=%2d' % \
               ( authors[i][0], spacetime_Z[i,0], no_papers[i] ) )

    print( 'E[sne]=',        sne_E )
    print( 'E[tsne]=',       tsne_E )
    print( 'E[spacetime] =', spacetime_E )

    #for alg,Y in [ ('sne',sne_Y), ('tsne',tsne_Y), ('spacetime',spacetime_Y) ]:
    #    d2 = dist2.dist2( Y )
    #    radius = np.sqrt( d2.max(1) ).mean()
    #    print( '[%s] co-author distance=%.3f' % ( alg,
    #             np.sqrt( d2 * (C > 0) ).mean() / radius ) )

    #scale_no = ( no_papers - no_papers.min() ) / ( no_papers.max() - no_papers.min() )
    #visualize( tsne_y,      authors, C, scale_no, big_guys, 't-SNE' )
    #visualize( tsne_multi_y[0], authors, C, no_papers, big_guys, 't-SNE_0' )
    #visualize( tsne_multi_y[1], authors, C, no_papers, big_guys, 't-SNE_1' )
    #visualize( spacetime_multi_y[0], authors, C, spacetime_multi_w[0], big_guys, 'spacetime_0' )
    #visualize( spacetime_multi_y[1], authors, C, spacetime_multi_w[1], big_guys, 'spacetime_1' )

    fig_file = os.path.splitext( result_file )[0] + '.pdf'
    __visualize( spacetime_Y, spacetime_Z, authors, no_papers, fig_file )

