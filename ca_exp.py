#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import os, re, gzip

EPS = np.finfo( float ).eps

def __rebuild( data_file, bin_file, dtype ):
    if data_file.endswith( ".gz" ):
        graphfile = gzip.open( data_file )
    else:
        graphfile = open( data_file )

    nodes = []
    edges = []
    linkre = re.compile( r'^\s*(\d+)\s+(\d+)\s*$' )
    try:
        for line in graphfile:
            if linkre.match( line ):
                a, b = sorted( [ int(n) for n in
                                 linkre.match( line ).groups() ] )

                if not a in nodes: nodes.append( a )
                if not b in nodes: nodes.append( b )

                if a == b:
                    print( '[ignored]', line, end="" )
                elif not ( (a,b) in edges ):
                    edges.append( (a,b) )
            else:
                print( '[ignored]', line, end="" )

    finally:
        graphfile.close()

    print( '%d nodes %d edges' % ( len(nodes), len(edges) ) )

    nodeidx = { node : idx for idx, node in enumerate( nodes ) }
    P = np.zeros( [len(nodes), len(nodes)], dtype=dtype )
    for a, b in edges:
        P[nodeidx[a],nodeidx[b]] += 1
    P = P + P.T

    idx = ( P.sum(0) > 0 )
    print( 'removing %d authors with no collaboration' % \
           ( len(nodes) - idx.sum() ) )
    P = P[idx][:,idx]

    P /= P.sum(0)
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, EPS )
    print( 'coauthor matrix of size', P.shape )

    print( "saving to '%s'" % bin_file )
    np.savez( bin_file, P=P )

def load_ca( data_file='data/ca-GrQc.txt.gz', dtype=np.float32 ):
    root, ext = os.path.splitext( data_file )
    if ext in [ '.gz', '.bz2' ]:
       root = os.path.splitext( root )[0]
    bin_file = root + '.npz'

    if not os.access( data_file, os.R_OK ):
        raise RuntimeError( "'%s' missing" % data_file )

    if not os.access( bin_file, os.R_OK ):
        __rebuild( data_file, bin_file, dtype )

    print( 'loading from %s' % bin_file )
    _tmp = np.load( bin_file )
    return _tmp['P']

if __name__ == '__main__':
    pass

