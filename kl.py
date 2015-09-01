#!/usr/bin/env python

from __future__ import print_function

from nips import load_nips
from words import load_words

import spacetime
import numpy as np
import scipy.io
import itertools
import sys, os, re, argparse

DATA_ARR  = [ 'nips22', 'words1000', 'words5000' ]
ALG_ARR   = [ 'sne', 'tsne', 'st' ]
DIM_ARR   = [ 2, 3, 4 ]
REPEAT    = 100
spacetime.conv_threshold = 1e-9
spacetime.min_epochs     = 500   # ensure enough convergence
INIT_SEED = 3129

CONFIGS = list( itertools.product( DATA_ARR, ALG_ARR, DIM_ARR ) )

parser = argparse.ArgumentParser()
parser.add_argument( 'exp', choices=range(len(CONFIGS)), type=int )
args = parser.parse_args()

DATA, ALG, DIM = CONFIGS[ args.exp ]

nips_pattern = re.compile( '^nips(\d+)$' )
words_pattern = re.compile( '^words(\d+)$' )
if nips_pattern.match( DATA ):
    num_vols = int( nips_pattern.match( DATA ).groups()[0] )
    C, P, authors, no_papers = load_nips( num_vols )
    print( "%d authors" % C.shape[0] )

elif words_pattern.match( DATA ):
    num_words = int( words_pattern.match( DATA ).groups()[0] )
    P, words = load_words( num_words )
    print( "%d words" % P.shape[0] )

else:
    print( 'unknown dataset: ', DATA )
    sys.exit(1)

if P.shape[0] > 2000: REPEAT /= 5

if ALG == 'sne':
    spacetime.distribution = 'gaussian'
    Y,Z,E = spacetime.st_snep( P,
                               DIM, 0,
                               repeat=REPEAT,
                               init_seed=INIT_SEED )

elif ALG == 'tsne':
    spacetime.distribution = 'student'
    Y,Z,E = spacetime.st_snep( P,
                               DIM, 0,
                               repeat=REPEAT,
                               init_seed=INIT_SEED )

elif ALG == 'st':
    spacetime.distribution = 'student'
    Y,Z,E = spacetime.st_snep( P,
                               DIM-1, 1,
                               repeat=REPEAT,
                               init_seed=INIT_SEED )

#elif ALG == 'multi-tsne':
#    tsne_multi_y,tsne_multi_w = tsne_multi.tsnep( P, 2, E=E )
#elif ALG == 'multi-st':
#    spacetime_y,spacetime_w = spacetime_multi.st_snep( P, 2, layers=2, E=E )

else:
    raise RuntimeError( 'unknown algorithm: %s' % ALG )

print( '[%10s %10s]  %.5f' % ( DATA, ALG, E ) )

