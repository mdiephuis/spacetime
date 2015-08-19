#!/usr/bin/env python

'''
Multiple Space Time Embedding (STE)
Ke Sun
University of Geneva
Summer 2013
'''

#try:
#    import ml
#    dist2 = ml.dist2
#except ImportError:
import dist2 as md2
dist2 = md2.dist2
dist2xy = md2.dist2xy

import d2p, pca
import itertools
import numpy as np
import sys, time

opt = sys.modules[ __name__ ]

opt.max_epochs = 2000
opt.lrate = 500                    # fixed learning rate
opt.lrate_pi = .1
opt.lrate_delta = .1
opt.momentum_init = 0.5            # momentum
opt.momentum_final = 0.8
opt.momentum_switch_epoch = 250
opt.min_gain = 1e-2
opt.distribution = 'student'
opt.lying = 0 #100
opt.output_intv = 5
opt.conv_threshold = 1e-7
opt.eps = 1e-30

sys.stdout = sys.stderr

def converged( E ):
    if len( E ) < 10: return False
    if (max(E[-5:]) - min(E[-5:])) / np.abs( E[-1] - E[0] ) > opt.conv_threshold:
        return False

    return True

def st_sne( data, dim, layers=2, perplexity=30,
        verbose=True,
        E=[] ):
    '''Space-Time Embedding

    data is the NxDIM data matrix,
    dim is the embedding dimension (dim<<DIM),
    return the Nxdim embedding coordinates
    '''

    if data.shape[1] > 30:
        if verbose: print( 'PCA %d->%d' % (data.shape[1], 30) )
        data = pca.pca( data, 30 )
    return st_sned( dist2( data ), dim, layers, perplexity, verbose, E )

def st_sned( dx2, dim, layers=2, perplexity=30,
        verbose=True,
        E=[] ):

    # encoding
    if verbose: print "computing probabilities with perplexity=%d..." \
                    % perplexity
    P = d2p.d2p( dx2, perplexity )
    for i in range(dx2.shape[0]): P[i,i] = 0
    P = P + P.T
    P /= P.sum()
    P = np.maximum( P, opt.eps )
    if verbose: print "done"

    return st_snep( P, dim, layers, verbose, E )

def st_snep( P, dim, layers=2, verbose=True, E=[] ):

    N = P.shape[0]
    start_t = time.time()

    if verbose:
        print "---------------------------------------"
        print "Multi Space-Time Embedding"
        print "%-15s = %d" % ( "#layers", layers )
        print "%-15s = %d" % ( "N", N )
        print "%-15s = %d" % ( "dim(y)", dim )
        print "%-15s = %d" % ( "max_epochs", opt.max_epochs )
        print "%-15s = %.1f" % ( "learning rate", opt.lrate )
        print "%-15s = %.1f" % ( "lrate (pi)", opt.lrate_pi)
        print "%-15s = %.1f" % ( "momentum", opt.momentum_init )
        print "%-15s = %s" % ( "lying", opt.lying )
        print "---------------------------------------"

    # initialize
    Y = 1e-4 * np.random.randn( layers, N, dim )
    pi = np.zeros( [layers, N] )
    delta = 1e-4

    Y_incs = np.zeros( [layers, N, dim] )
    gains = np.ones( Y.shape )
    delta_incs = 0
    delta_gain = 1

    if opt.lying > 0:
        if verbose: print "[%4d]  lying with P=P*4" % 0
        P *= 4                    # the lying trick of van de maaten

    for epoch in range( opt.max_epochs ):
        if ( opt.lying > 0 ) and ( epoch == opt.lying ):
            if verbose: print "[%4d]  stop lying" % epoch
            P /= 4

        w = np.exp( pi )
        w /= w.sum( 0 )

        sim = np.zeros( [N, N] )
        for k, l in list( itertools.product(*[range(layers), range(layers)]) ):
            sim += ( w[k][:,np.newaxis] * w[l] ) \
                    * np.exp( (k-l)**2 * delta ) \
                    / ( 1 + dist2xy(Y[k], Y[l]) )

        Q = sim.copy()
        for i in range( N ): Q[i,i] = 0
        Q /= Q.sum()
        Q = np.maximum( Q, opt.eps )
        E.append( (P*np.log(P)).sum() - (P*np.log(Q)).sum() )

        # gradient
        pi_grad = np.zeros( pi.shape )
        Y_grad = np.zeros( Y.shape )
        delta_grad = 0

        A = (P-Q) / sim

        for k, l in list( itertools.product(*[range(layers), range(layers)]) ):
            AB = A * np.exp( (k-l)**2 * delta ) / ( 1 + dist2xy(Y[k], Y[l]) )
            pi_grad[k] -= 2 * np.dot( AB, w[l] )

            W = (w[k][:,np.newaxis] * w[l]) * AB / ( 1 + dist2xy(Y[k], Y[l]) )
            Y_grad[k] += 4 * ( np.dot( np.diag(W.sum(1)), Y[k] ) - np.dot( W, Y[l] ) )

            if k < l:
                print "factor", np.dot( w[k], np.dot(AB, w[l]) )
                #delta_grad -= 2 * np.dot( w[k], np.dot(AB, w[l]) ) * (k-l)**2
                delta_grad -= (w[k][:,np.newaxis] * w[l] * AB ).sum()

        pi_grad *= w
        pi -= opt.lrate_pi * pi_grad

        if epoch < opt.momentum_switch_epoch:
            momentum = opt.momentum_init
        else:
            momentum = opt.momentum_final

        if np.sign( delta_incs ) != np.sign( delta_grad ):
            delta_gain += .2
        else:
            delta_gain *=.8
        delta_gain = np.maximum( delta_gain, opt.min_gain )
        delta_incs = momentum * delta_incs - opt.lrate_delta * delta_gain * np.sign( delta_grad )
        delta += delta_incs

        gains = (gains+.2) * ( np.sign(Y_grad)!=np.sign(Y_incs) ) \
              + (gains*.8) * ( np.sign(Y_grad)==np.sign(Y_incs) )
        gains = np.maximum( gains, opt.min_gain )
        Y_incs = momentum * Y_incs - opt.lrate * gains * Y_grad
        Y += Y_incs

        conv_flag = converged( E )
        if conv_flag or epoch % opt.output_intv == 0:
            if verbose: print "[%4d] " % epoch,
            if verbose: print "|Y|=%5.2f  " % np.mean( np.abs(Y) ),
            if verbose: print "|Y_grad|=%8.4fe-5  " % \
                              (np.mean(np.abs(Y_grad))*1e5),
            if verbose: print "|pi_grad|=%8.4fe-3 " % \
                              (np.mean(np.abs(pi_grad))*1e3),

            if verbose: print "delta_grad=%.4f" % (delta_grad*1e10),
            if verbose: print "delta=%.4f" % delta,

            if verbose: print "E=%6.3f " % E[-1]
        if conv_flag: break

    if verbose:
        if epoch < opt.max_epochs-1:
            print "converged after %d epochs" % epoch
        total_t = int( time.time() - start_t )
        print "total running time is %02dh:%02dm" % \
            ( total_t/3600, (total_t%3600)/60 )

    w = np.exp( pi )
    w /= w.sum( 0 )

    return Y, w
