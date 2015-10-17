from __future__ import print_function
from __future__ import division

from time import time, strftime, gmtime

def tick( start_t=None, verbose=True ):
    current_t = time()

    if start_t is None:
        if verbose: print( strftime( "%a %b-%d %H:%M", gmtime() ) )
    else:
        taken = current_t - start_t
        if verbose:
            if taken < 3600:
                print( '%2dm%2ds' % ( (taken//60), (taken%60) ) )
            else:
                print( '%2dh%2dm' % ( (taken//3600), (taken%3600)//60 ) )

    return current_t

