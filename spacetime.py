from __future__ import print_function

"""
Space-time Embedding

Batch gradient descent optimizer: along the space-like dimensions
the delta-bar-delta rule (t-SNE by van der Maaten) is used; along
the time-like dimensions a global adaptive learning rate is used

Ke Sun
University of Geneva
Summer 2013 - Summer 2015
"""

try:
    import theano
    X = theano.tensor.fmatrix( 'X' )
    diff = X.reshape( (X.shape[0], 1, -1) ) \
         - X.reshape( (1, X.shape[0], -1) )
    square_distances = (diff ** 2).sum(2)
    dist2 = theano.function( [X], square_distances )
except:
    from dist2 import dist2

from tick import tick
import d2p
import numpy as np
import sys

opt = sys.modules[ __name__ ]               # global constants
opt.min_epochs            = 50              # min no of iterations
opt.max_epochs            = 5000            # max no of iterations
opt.lrate_s               = 100             # learning rate along space
opt.lrate_t               = 1               # learning rate along time
opt.momentum              = (0.5, 0.8, 0.9) # momentums
opt.momentum_epoch        = (250, 1000)     # at which epoch to speed up
opt.distribution          = 'student'
opt.lying                 = 100
opt.output_intv           = 100
opt.conv_threshold        = 1e-7
opt.dtype                 = np.float32
opt.eps                   = np.finfo( float ).eps

def st_sne( data,
            dim_s,
            dim_t,
            perplexity=30,
            init_y=None,
            verbose=True,
            repeat=1,
            init_seed=None ):
    '''Space-Time Embedding (based on raw features)

    data is the NxDIM data matrix,
    dim is the embedding dimension (dim<<DIM),
    return the Nxdim embedding coordinates
    '''

    #if data.shape[1] > 30:
    #    if verbose: print( 'PCA %d->%d' % (data.shape[1], 30) )
    #    data = pca.pca( data, 30 )

    return st_sned( dist2( data ), dim_s, dim_t, \
                    perplexity, init_y, verbose, repeat, init_seed )

def st_sned( dx2,
             dim_s,
             dim_t,
             perplexity=30,
             init_y=None,
             verbose=True,
             repeat=1,
             init_seed=None ):
    '''
    Space-time Embedding (based on pair-wise distance)
    '''

    # converting dx2 into a probability matrix
    if verbose: print( "computing probabilities with perplexity=%d..." \
                    % perplexity, end="" )
    P = d2p.d2p( dx2, perplexity )
    for i in range( dx2.shape[0] ): P[i,i] = 0
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, opt.eps )
    if verbose: print( "done" )

    return st_snep( P, dim_s, dim_t, init_y, verbose, repeat, init_seed )

def st_snep( P,
             dim_s,
             dim_t,
             init_y=None,
             verbose=True,
             repeat=1,
             init_seed=None ):
    '''
    Space-time Embedding (based on pair-wise similarities)
    '''

    if verbose:
        print( "Space-Time Embedding (%s)" % opt.distribution )
        print( "%-15s = %d"   % ( "N", P.shape[0] ) )
        print( "%-15s = %d"   % ( "dim(s)", dim_s ) )
        print( "%-15s = %d"   % ( "dim(t)", dim_t ) )
        print( "%-15s = %d"   % ( "max_epochs", opt.max_epochs ) )
        print( "%-15s = %.1f" % ( "lrate (s)", opt.lrate_s ) )
        print( "%-15s = %.1f" % ( "lrate (t)", opt.lrate_t ) )
        print( "%-15s = %d"   % ( "lying", opt.lying ) )

    best_Y = None
    best_Z = None
    best_E = np.inf

    if init_seed is None: init_seed = np.random.randint( 1000000 )

    start_t = tick( verbose=verbose )

    for run in range( repeat ):
        Y, Z, E = __embed( P, dim_s, dim_t, init_y, verbose, init_seed+run )

        if E[-1] < best_E:
            best_E = E[-1]
            best_Y = Y
            best_Z = Z

        if verbose:
            tick( start_t )
            print( 'final  E=%6.3f' % E[-1] )
            print( 'record E=%6.3f' % best_E )

    return best_Y, best_Z, best_E

def __init( N, dim_s, dim_t, init_y, seed ):
    '''
    random initialization
    '''

    if seed is None:
        chaos = np.random.RandomState()
    else:
        chaos = np.random.RandomState( seed )

    if init_y is None:
        Y = 1e-4 * chaos.randn( N, dim_s ).astype( opt.dtype )
    else:
        Y = init_y

    Z = 1e-5 * chaos.randn( N, dim_t ).astype( opt.dtype )

    return Y, Z

def __momentum( epoch ):
    if epoch < opt.momentum_epoch[0]:
        m = opt.momentum[0]
    elif epoch < opt.momentum_epoch[1]:
        m = opt.momentum[1]
    else:
        m = opt.momentum[2]
    return m

def __converged( E ):
    '''
    determine convergence
    '''
    if len( E ) < opt.min_epochs: return False

    return ( (max(E[-5:]) - min(E[-5:]))
             / np.abs( E[-1] - E[0] )
             < opt.conv_threshold )

def __embed( P, dim_s, dim_t, init_y, verbose, seed ):
    '''
    obtain an embedding based on batch gradient descent
    '''

    N = P.shape[0]

    Y, Z    = __init( N, dim_s, dim_t, init_y, seed )
    Y_incs  = np.zeros_like( Y )
    Y_gains = np.ones_like(  Y )
    Z_incs  = np.zeros_like( Z )
    Z_gain  = 1
    E       = []

    if opt.lying > 0:
        if verbose: print( "[%4d]  lying with P=P*4" % 0 )
        P *= 4      # the lying trick, also known as "early exaggeration"

    for epoch in range( opt.max_epochs ):
        momentum = __momentum( epoch )

        if ( opt.lying > 0 ) and ( epoch == opt.lying ):
            if verbose: print( '[%4d]  stop lying' % epoch )
            P /= 4

        dy2 = dist2( Y )
        if dim_t > 0:
            dz2 = dist2( Z )
        else:                # without time-dimension
            dz2 = 0

        if opt.distribution == 'student':
            Q = np.exp( dz2 ) / ( 1 + dy2 )
        else:
            Q = np.exp( dz2 - dy2 )
        for i in range( N ): Q[i,i] = 0
        Q /= ( Q.sum() + 10 * opt.eps )
        Q = np.maximum( Q, opt.eps )

        E.append( (P*np.log(P)).sum() - (P*np.log(Q)).sum() )
        if not np.isfinite( E[-1] ): break

        if opt.distribution == 'student':
            W_y = ( P - Q ) / ( 1 + dy2 )
        else:
            W_y = ( P - Q )
        Y_grad = np.dot( np.diag( W_y.sum(0) ) - W_y, Y )
        Y_grad -= Y_grad.mean( 0 )

        Y_gains = (Y_gains+.2) * ( np.sign(Y_grad)!=np.sign(Y_incs) ) \
                + (Y_gains*.8) * ( np.sign(Y_grad)==np.sign(Y_incs) )

        Y_incs *= momentum
        Y_incs -= opt.lrate_s * Y_gains * Y_grad
        Y += Y_incs
        Y -= Y.mean(0)

        # learn slower on time-like dimensions
        if dim_t > 0:
            W_z = ( Q - P )
            Z_grad = np.dot( np.diag( W_z.sum(0) ) - W_z, Z )
            Z_grad -= Z_grad.mean( 0 )

            if (Z_grad * Z_incs).sum() < 0:
                Z_gain += .2
            else:
                Z_gain *= .8

            Z_incs *= momentum
            Z_incs -= opt.lrate_t * Z_gain * Z_grad
            Z += Z_incs
            Z -= Z.mean(0)

        conv_flag = __converged( E )
        if verbose and ( conv_flag or epoch % opt.output_intv == 0 ):
            print( "[%4d] |Y|=%6.2f |Y_grad|=%7.3fe-5" \
                   % ( epoch,
                       np.abs(Y).max(),
                       np.mean(np.abs(Y_grad))*1e5 ), end=" " )

            if dim_t > 0:
                print( "|Z|=%5.2f |Z_grad|=%7.3fe-5" \
                       % ( np.abs(Z).max(),
                           np.mean(np.abs(Z_grad))*1e5 ), end=" " )

            if conv_flag:
                print( "E=%6.3f (converged)" % E[-1] )
            else:
                print( "E=%6.3f" % E[-1] )

        if conv_flag: break

    return Y, Z, E

