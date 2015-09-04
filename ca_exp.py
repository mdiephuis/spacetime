#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import os, sys, re, gzip

EPS = np.finfo( float ).eps

def load_ca( filename='data/ca-GrQc.txt.gz' ):
    if not os.access( filename, os.R_OK ):
        raise RuntimeError( "no such file '%s'" % filename )

    if filename.endswith( ".gz" ):
        graphfile = gzip.open( filename )
    else:
        graphfile = open( filename )

    nodes = []
    edges = []
    linkre = re.compile( r'^\s*(\d+)\s*(\d+)\s*$' )
    try:
        for line in graphfile:
            if linkre.match( line ):
                a, b = sorted( [ int(n) for n in linkre.match( line ).groups() ] )

                if not a in nodes: nodes.append( a )
                if not b in nodes: nodes.append( b )

                if a == b:
                    print( '[ignored] self connection of %d' % a )
                elif not ( (a,b) in edges ):
                    edges.append( (a,b) )
            else:
                print( '[ignored]', line, end="" )

    finally:
        graphfile.close()

    print( '%d nodes %d edges' % ( len(nodes), len(edges) ) )

    nodeidx = { node : idx for idx, node in enumerate( nodes ) }
    P = np.zeros( [len(nodes), len(nodes)], dtype=np.float32 )
    for a, b in edges:
        P[nodeidx[a],nodeidx[b]] += 1
    P = P + P.T

    idx = ( P.sum(0) > 0 )
    print( 'removing %d authors with no collaboration' % \
           ( len(nodes) - idx.sum() ) )
    P = P[idx][:,idx]

    P /= P.sum(1)[:,np.newaxis]
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, EPS )
    print( P.shape )

    return P

if __name__ == '__main__':
    pass
