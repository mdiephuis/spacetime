#!/usr/bin/env python

from __future__ import print_function

from school_exp import load_school
from ca_exp import load_ca
from nips_exp import load_nips
from words_exp import load_words

import spacetime
import numpy as np
import scipy.io
import itertools
import sys, os, re, argparse

def load_data( data ):
    nips_pattern  = re.compile( '^nips(\d+)$' )
    words_pattern = re.compile( '^words(\d+)$' )

    if data == 'school':
        P = load_school()

    elif data == 'ca':
        P = load_ca()

    elif nips_pattern.match( data ):
        num_vols = int( nips_pattern.match( data ).groups()[0] )
        C, P, authors, no_papers = load_nips( num_vols )

    elif words_pattern.match( data ):
        num_words = int( words_pattern.match( data ).groups()[0] )
        P, words = load_words( num_words )

    else:
        raise RuntimeError( 'unknown dataset: %s' % data )

    print( 'input shape:', P.shape )
    return P

def embed( P, alg, dim, repeat, seed ):
    if P.shape[0] > 2000: repeat /= 5

    if alg == 'sne':
        spacetime.distribution = 'gaussian'
        Y,Z,E = spacetime.st_snep( P,
                                   dim, 0,
                                   repeat=repeat,
                                   init_seed=seed )

    elif alg == 'tsne':
        spacetime.distribution = 'student'
        Y,Z,E = spacetime.st_snep( P,
                                   dim, 0,
                                   repeat=repeat,
                                   init_seed=seed )

    elif alg == 'st':
        spacetime.distribution = 'student'
        Y,Z,E = spacetime.st_snep( P,
                                   dim-1, 1,
                                   repeat=repeat,
                                   init_seed=seed )

    #elif ALG == 'multi-tsne':
    #    tsne_multi_y,tsne_multi_w = tsne_multi.tsnep( P, 2, E=E )
    #elif ALG == 'multi-st':
    #    spacetime_y,spacetime_w = spacetime_multi.st_snep( P, 2, layers=2, E=E )

    else:
        raise RuntimeError( 'unknown algorithm: %s' % alg )

    return E

if __name__ == '__main__':
    DATA_ARR  = [ 'ca', 'nips17', 'nips22', 'words1000', 'words5000' ]
    #DATA_ARR  = [ 'school' ]
    ALG_ARR   = [ 'sne', 'tsne', 'st' ]
    DIM_ARR   = [ 2, 3, 4 ]
    REPEAT    = 100
    SEED      = 0

    # pre-compile all dataset
    for data in DATA_ARR: load_data( data )

    # ensure enough convergence
    spacetime.conv_threshold = 1e-9
    spacetime.min_epochs     = 500
    spacetime.lrate_s = 10
    spacetime.lrate_t = .1

    CONFIGS = list( itertools.product( DATA_ARR, ALG_ARR, DIM_ARR ) )
    for idx, (data, alg, dim) in enumerate( CONFIGS ):
        print( '%2d: data=%-10s alg=%-6s dim=%-2d' % ( idx, data, alg, dim ) )

    parser = argparse.ArgumentParser()
    parser.add_argument( 'exp', choices=range(len(CONFIGS)), type=int )
    args = parser.parse_args()

    DATA, ALG, DIM = CONFIGS[ args.exp ]
    P = load_data( DATA )
    E = embed( P, ALG, DIM, REPEAT, SEED )

    print( '[%10s %10s]  %.5f' % ( DATA, ALG, E ) )

