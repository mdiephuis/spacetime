from __future__ import print_function
import numpy as np
import warnings

MAX_TRY = 100
TOL     = 1e-3
EPS     = np.finfo( float ).eps

def d2p( d2, perplexity, beta=None, exclude_self=True, verbose=False ):
    '''Convert a square-distance matrix to a probability matrix
       with Gaussian kernels

       d2 is a NxN distance matrix,
       perplexity is the effective number of NNs (1<perplexity<N),
       beta is the output multipliers to the rows of d2,
       verbose enables console output,

       return the NxN matrix { p_{j|i} }, p_{j|i} is the probability
       of selecting point j as a neighbour of point i (i,j=1...N)
       Note the resulting P matrix is not symmetric
    '''

    if not (d2.ndim == 2 and d2.dtype == np.float32) :
        raise RuntimeError( "data should be a float32 matrix" )

    N,n = d2.shape
    if exclude_self and (N != n):
        raise RuntimeError( "data should be a square matrix" )

    if perplexity < 2:
        warnings.warn( "perplexity should be at least 2", RuntimeWarning )
        perplexity = np.maximum( perplexity, 2 )

    elif perplexity > n-2:
        warnings.warn( "perplexity should be at most %d" % (n-2),
                       RuntimeWarning )
        perplexity = np.minimum( perplexity, n-2 )

    if verbose:
        print( "computing P(%dx%d) with per=%.0f..." % (N, n, perplexity), )

    P = np.zeros_like( d2 )
    if beta is None: beta = np.ones( N )

    for i in range( N ):
        # start with a very big perplexity
        beta[i] = 1. / d2[i].max()
        betamin = -np.inf
        betamax = np.inf

        for tries in range( MAX_TRY ):
            p = np.exp( -d2[i] * beta[i] )
            if exclude_self: p[i] = 0

            E = np.log( p.sum() ) + beta[i]*(p*d2[i]).sum() / ( p.sum()+10*EPS )
            Ediff = E - np.log( perplexity )
            if( np.abs(Ediff) < TOL ):
                break

            elif Ediff > 0:
                betamin = beta[i]
                if betamax == np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2

            else:
                betamax = beta[i]
                if betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

        # recheck
        E = np.log(p.sum()) + beta[i]*(p*d2[i]).sum() / (p.sum()+10*EPS)
        if np.abs( E - np.log(perplexity) ) > TOL:
            print( "perplexity=%.1f does not satisfy the" \
                   "requirement perplexity=%.0f" % (np.exp(E), perplexity) )
            print( "sample [%d]" % i )
            print( "%d tries" % tries )
            print( "beta=%f" % beta[i] )

        P[i] = p / p.sum()

    if verbose:
        print( 'done' )
        print( 'mean(sigma) = %7.4f' % (np.sqrt(.5/beta)).mean() )
        print( ' min(sigma) = %7.4f' % (np.sqrt(.5/beta)).min() )
        print( ' max(sigma) = %7.4f' % (np.sqrt(.5/beta)).max() )

    return P

