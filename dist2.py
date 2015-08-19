import numpy as np
import warnings

def dist2( data ):
    '''Compute pair-wise square distances

    data is a NxD data matrix,
    return the NxN square distances
    '''
    #warnings.warn( "using the slow python implementation of dist2()", \
    #               DeprecationWarning )
    if data.ndim != 2: raise RuntimeError( "data should be a 2D array" )

    xy = np.dot( data, data.T )
    x2 = np.diag( xy ).copy()[:,np.newaxis]

    return np.maximum( -2*xy + x2 + x2.T, 0 )
