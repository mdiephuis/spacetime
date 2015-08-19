#!/usr/bin/env python

from __future__ import print_function

"""
Space-time Embedding

the optimizer is a batch gradient descent
using the delta-bar-delta rule, and is
mostly based on the t-SNE optimizer by
Laurens van der Maaten

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

import d2p
import numpy as np
import sys, time

opt = sys.modules[ __name__ ]

opt.min_epochs            = 0     # min no of iterations
opt.max_epochs            = 3000  # max no of iterations
opt.lrate_s               = 100   # learning rate along space
opt.lrate_t               = 1     # learning rate along time
opt.momentum_init         = 0.5   # initial momentum
opt.momentum_final        = 0.8   # final momentum
opt.momentum_switch_epoch = 250   # when to speed up
opt.min_gain              = 1e-3
opt.max_gain              = 10
opt.distribution          = 'student'
opt.lying                 = 100
opt.output_intv           = 100
opt.conv_threshold        = 1e-7
opt.eps                   = np.finfo( float ).eps

sys.stdout = sys.stderr

def converged( E ):
    '''
    determine convergence
    '''
    if len( E ) < 10: return False
    if len( E ) < opt.min_epochs: return False

    return ( (max(E[-5:]) - min(E[-5:])) \
             / np.abs( E[-1] - E[0] ) \
             < opt.conv_threshold )

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
                    % perplexity )
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

    N = P.shape[0]
    start_t = time.time()

    if verbose:
        print( "Space-Time Embedding (%s)" % opt.distribution )
        print( "%-15s = %d"   % ( "N", N ) )
        print( "%-15s = %d"   % ( "dim(s)", dim_s ) )
        print( "%-15s = %d"   % ( "dim(t)", dim_t ) )
        print( "%-15s = %d"   % ( "max_epochs", opt.max_epochs ) )
        print( "%-15s = %.1f" % ( "learning rate (s)", opt.lrate_s ) )
        print( "%-15s = %.1f" % ( "learning rate (t)", opt.lrate_t ) )
        print( "%-15s = %.1f" % ( "momentum (init)", opt.momentum_init ) )
        print( "%-15s = %.1f" % ( "momentum (final)", opt.momentum_final ) )
        print( "%-15s = %s"   % ( "lying", opt.lying ) )

    best_Y = None
    best_Z = None
    best_E = np.inf

    for run in range( repeat ):
        if not ( init_seed is None ): np.random.seed( init_seed + run )

        # initialize
        if init_y is not None:
            Y = init_y
        else:
            Y = 1e-4 * np.random.randn( N, dim_s ).astype( np.float32 )
        Y_incs   = np.zeros_like( Y )
        Y_gains  = np.ones_like(  Y )

        Z = 1e-5 * np.random.randn( N, dim_t ).astype( np.float32 )
        Z_incs   = np.zeros_like( Z )
        Z_gains  = np.ones_like(  Z )

        E = []

        if opt.lying > 0:
            if verbose: print( "[%4d]  lying with P=P*4" % 0 )
            P *= 4      # the lying trick, also known as
                        # "early exaggeration"

        for epoch in range( opt.max_epochs ):
            if epoch < opt.momentum_switch_epoch:
                momentum = opt.momentum_init
            else:
                momentum = opt.momentum_final

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
            #elif opt.distribution == 'student_t':
            #    Q = ( 1 + dz2 ) / ( 1 + dy2 )
            else:
                Q = np.exp( dz2 - dy2 )
            for i in range( N ): Q[i,i] = 0
            Q /= ( Q.sum() + 10 * opt.eps )
            Q = np.maximum( Q, opt.eps )
            E.append( (P*np.log(P)).sum() - (P*np.log(Q)).sum() )

            if opt.distribution == 'student':
                W_y = ( P - Q ) / ( 1 + dy2 )
            else:
                W_y = ( P - Q )
            Y_grad = np.dot( np.diag( W_y.sum(0) ) - W_y, Y )
            Y_grad -= Y_grad.mean( 0 )

            Y_gains = (Y_gains+.2) * ( np.sign(Y_grad)!=np.sign(Y_incs) ) \
                    + (Y_gains*.8) * ( np.sign(Y_grad)==np.sign(Y_incs) )
            Y_gains = np.maximum( Y_gains, opt.min_gain )
            Y_incs = momentum * Y_incs - opt.lrate_s * Y_gains * Y_grad
            Y += Y_incs
            Y -= Y.mean(0)

            if dim_t > 0:
                #if opt.distribution == 'student_t':
                #    W_z = ( Q - P ) / ( 1 + dz2 )
                W_z = ( Q - P )
                Z_grad = np.dot( np.diag( W_z.sum(0) ) - W_z, Z )
                Z_grad -= Z_grad.mean( 0 )

                # learning slower on time-like dimensions
                Z_gains = (Z_gains+.2) * ( np.sign(Z_grad)!=np.sign(Z_incs) ) \
                        + (Z_gains*.8) * ( np.sign(Z_grad)==np.sign(Z_incs) )
                Z_gains = np.maximum( Z_gains, opt.min_gain )
                Z_gains = np.minimum( Z_gains, opt.max_gain )
                Z_incs = momentum * Z_incs - opt.lrate_t * Z_gains * Z_grad
                #Z_incs = momentum * Z_incs - opt.lrate_t * Z_grad

                Z += Z_incs
                Z -= Z.mean(0)

            conv_flag = converged( E )
            if verbose and ( conv_flag or epoch % opt.output_intv == 0 ):
                print( "[%4d] |Y|=%5.2f |Y_grad|=%8.4fe-5 " \
                       % ( epoch,
                           np.abs(Y).max(),
                           np.mean(np.abs(Y_grad))*1e5 ), end="" )

                if dim_t > 0:
                    print( "|Z|=%5.2f |Z_grad|=%8.4fe-5" \
                           % ( np.abs(Z).max(),
                               np.mean(np.abs(Z_grad))*1e5 ), end="" )

                if conv_flag:
                    print( "E=%6.3f (converged)" % E[-1] )
                else:
                    print( "E=%6.3f" % E[-1] )

            if conv_flag: break

        if E[-1] < best_E:
            best_E = E[-1]
            best_Y = Y
            best_Z = Z

        if verbose:
            total_t = int( time.time() - start_t )
            print( "running time is %02dh:%02dm" % \
                    ( total_t/3600, (total_t%3600)/60 ) )

            print( 'final  E=%6.3f'  % E[-1] )
            print( 'record E=%6.3f' % best_E )

    return best_Y, best_Z, best_E

