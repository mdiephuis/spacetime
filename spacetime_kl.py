#!/usr/bin/env python

from __future__ import print_function

import spacetime, spacetime_nips, spacetime_words
import numpy as np
import scipy.io
import itertools
import sys, os
import re

DATA_ARR  = [ 'nips', 'words1000', 'words5000' ]
ALG_ARR   = [ 'sne', 'tsne', 'st' ]
DIM_ARR   = [ 2, 3, 4 ]
REPEAT    = 50
spacetime.conv_threshold = 1e-9
spacetime.min_epochs     = 500   # ensure enough convergence
INIT_SEED = 3127

CONFIGS = list( itertools.product( DATA_ARR, ALG_ARR, DIM_ARR ) )

if len( sys.argv ) < 2: 
    print( 'usage: %s expno (0-%d)' % ( sys.argv[0], len(CONFIGS)-1 ) )
    sys.exit(1)

try:
    DATA, ALG, DIM = CONFIGS[ int(sys.argv[1]) ]
except:
    print( 'usage: %s expno (0-%d)' % ( sys.argv[0], len(CONFIGS)-1 ) )
    sys.exit(1)

if DATA == 'nips':
    C, P, authors, no_papers = spacetime_nips.load_nips()
    print( "%d authors" % C.shape[0] )

elif DATA.startswith( 'words' ):
    pattern = re.compile( '^words(\d+)$' )
    P, words = spacetime_words.load_words( 
               int( pattern.match( DATA ).groups()[0] ) )
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

