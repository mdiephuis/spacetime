#!/usr/bin/env python

from __future__ import print_function

import dist2, spacetime
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.transforms import Bbox
import numpy as np
import scipy.io
import os, itertools

REPEAT = 50
EPS    = np.finfo( float ).eps
OUTPUT = 'eps'

def __rebuild( data_file, bin_file, num_vols, dtype ):
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

def load_nips( num_vols, dtype=np.float32 ):
    '''load the NIPS co-authorship dataset'''

    if not ( num_vols in (17,22) ):
        raise RuntimeError( 'num_vols=%d' % num_vols )

    data_file = 'data/nips_1-%d.mat' % num_vols
    bin_file  = os.path.splitext( data_file )[0] + '.npz'

    if not os.access( data_file, os.R_OK ):
        raise RuntimeError( "'%s' missing" % data_file )

    if not os.access( bin_file, os.R_OK ):
        __rebuild( data_file, bin_file, num_vols, dtype )

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

def visualize( y, authors, C, weights, big_guys, title ):
    if OUTPUT == 'X':
        fig = plt.figure( figsize=[10,8] )
        fig.suptitle( title )
        ax = fig.add_subplot( 111 )
    else:
        fig = plt.figure( figsize=[10,8], frameon=True )
        ax = plt.Axes( fig, [0., 0., 1., 1.] )
        #ax.set_axis_off()
        fig.add_axes( ax )

    ax.scatter( y[:,0], y[:,1], s=15, linewidths=0, c='gray' )
    sc=ax.scatter( y[big_guys,0],
                   y[big_guys,1],
                   s=50,
                   c=weights[big_guys],
                   linewidths=.5,
                   cmap='RdYlGn',
                   vmin=-np.abs( weights ).max(),
                   vmax= np.abs( weights ).max() )
    cbar = plt.colorbar( sc )
    cbar.ax.set_aspect( 40 )

    x_min = y[big_guys,0].min()
    x_max = y[big_guys,0].max()
    x_gap = .02 * (x_max-x_min)
    y_min = y[big_guys,1].min()
    y_max = y[big_guys,1].max()
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
    font_s = np.abs(weights)*10 + 6    # 6pt - 16pt
    text_positions = y[big_guys] + np.array( [offset,0] )

    past_bb = []
    for i in range( len(big_guys) ):
        _x = text_positions[i][0]
        _y = text_positions[i][1]
        _a = authors[big_guys[i]][0].split('_')[0]
        tt = ax.text( _x,
                      _y,
                      _a,
                      size=font_s[big_guys[i]],
                      rotation=0,
                      color='black' )

        transf = ax.transData.inverted()
        bb = tt.get_window_extent( renderer = find_renderer( fig ) ).transformed( transf ).get_points()
        if overlap_ratio( past_bb, bb ) > .02:
            best_o      = np.inf
            best_adjust = None
            search_range = np.vstack( [ np.linspace( 0,  4*offset, 20 ),
                                        np.linspace( 0, -4*offset, 20 ) ]
                                    ).flatten('F')
            for x_adjust in search_range:
                for y_adjust in search_range:
                    oratio = overlap_ratio( past_bb, bb + np.array([x_adjust,y_adjust]) )
                    if oratio < best_o * .999:
                        best_o = oratio
                        best_adjust = np.array([x_adjust,y_adjust])
            bb += best_adjust
            tt.set_position( (bb[0,0], bb[0,1] ) )
        past_bb.append( bb )

    fig.savefig( '%s.pdf' % title,
                 bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=600 )

if __name__ == '__main__':
    num_vols = 22
    C, P, authors, no_papers = load_nips( num_vols )
    print( "%d authors" % C.shape[0] )
    big_guys = np.nonzero( no_papers >= 13 )[0]
    print( "%d authors have >=10 NIPS papers" % big_guys.size )

    # embedding
    result_file = 'spacetime_nips%d_result.npz' % num_vols
    if os.access( result_file, os.R_OK ):
        print( 'loading results from %s' % result_file )
        tmp = np.load( result_file )
        sne_E  = tmp['sne_E']
        sne_Y  = tmp['sne_Y']
        tsne_E = tmp['tsne_E']
        tsne_Y = tmp['tsne_Y']
        spacetime_E = tmp['spacetime_E']
        spacetime_Y = tmp['spacetime_Y']
        spacetime_Z = tmp['spacetime_Z']

        #tsne_multi_E = tmp['tsne_multi_E']
        #tsne_multi_y = tmp['tsne_multi_y']
        #tsne_multi_w = tmp['tsne_multi_w']
        #spacetime_multi_E = tmp['spacetime_multi_E']
        #spacetime_multi_y = tmp['spacetime_multi_y']
        #spacetime_multi_w = tmp['spacetime_multi_w']

    else:
        spacetime.conv_threshold = 1e-9
        spacetime.distribution   = 'gaussian'
        sne_Y,_tmp,sne_E   = spacetime.st_snep( P, 3, 0, repeat=REPEAT )

        spacetime.distribution = 'student'
        tsne_Y,_tmp,tsne_E = spacetime.st_snep( P, 3, 0, repeat=REPEAT )

        spacetime.distribution = 'student'
        spacetime_Y,spacetime_Z,spacetime_E = \
            spacetime.st_snep( P, 2, 1, repeat=REPEAT )

        #tsne_multi_E = []
        #tsne_multi_y,tsne_multi_w = tsne_multi.tsnep( P, 2, E=tsne_multi_E )
        #spacetime_multi_E = []
        #spacetime_multi_y,spacetime_multi_w = spacetime_multi.st_snep( P, 2, 2, E=spacetime_multi_E )

        np.savez( result_file, 
                  sne_E=sne_E,
                  sne_Y=sne_Y,
                  tsne_E=tsne_E,
                  tsne_Y=tsne_Y,
                  spacetime_E=spacetime_E,
                  spacetime_Y=spacetime_Y,
                  spacetime_Z=spacetime_Z,
                #tsne_multi_E=tsne_multi_E,
                #tsne_multi_y=tsne_multi_y,
                #tsne_multi_w=tsne_multi_w,
                #spacetime_multi_E=spacetime_multi_E,
                #spacetime_multi_y=spacetime_multi_y,
                #spacetime_multi_w=spacetime_multi_w,
                )

    # for single space time embedding
    scale_Z = np.sqrt( (spacetime_Z**2).sum(1) )
    scale_Y = np.sqrt( (spacetime_Y**2).sum(1) )
    #plt.hist( scale_z, 10 )

    rank = np.argsort( scale_Z )[::-1]
    print( 'top 10 authors by z:' )
    for i in rank[:10]:
        print( '%20s %7.3f' % ( authors[i][0], spacetime_Z[i,0] ) )

    rank_paper = np.argsort( no_papers )[::-1]
    print( 'top 10 authors by #papers:' )
    for i in rank_paper[:10]:
        print( '%20s %7.3f' % ( authors[i][0], no_papers[i] ) )

    print( 'E[sne]=',        sne_E )
    print( 'E[tsne]=',       tsne_E )
    print( 'E[spacetime] =', spacetime_E )

    for alg,Y in [('sne',sne_Y), ('tsne',tsne_Y), ('spacetime',spacetime_Y)]:
        d2 = dist2.dist2( Y )
        radius = np.sqrt( d2.max(1) ).mean()
        print( '[%s] co-author distance=%.3f' % ( alg,
                 np.sqrt( d2 * (C > 0) ).mean() / radius ) )

    #scale_no = ( no_papers - no_papers.min() ) / ( no_papers.max() - no_papers.min() )
    #visualize( tsne_y,      authors, C, scale_no, big_guys, 't-SNE' )
    visualize( spacetime_Y, authors, C, spacetime_Z,  big_guys, 'spacetime' )
    #visualize( tsne_multi_y[0], authors, C, no_papers, big_guys, 't-SNE_0' )
    #visualize( tsne_multi_y[1], authors, C, no_papers, big_guys, 't-SNE_1' )
    #visualize( spacetime_multi_y[0], authors, C, spacetime_multi_w[0], big_guys, 'spacetime_0' )
    #visualize( spacetime_multi_y[1], authors, C, spacetime_multi_w[1], big_guys, 'spacetime_1' )
    plt.show()
