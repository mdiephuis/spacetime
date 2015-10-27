#!/usr/bin/env python

import matplotlib
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import rc

rc( 'xtick', direction='out' )
rc( 'ytick', direction='out' )
rc( 'font', **{'family':'serif','serif':['Palatino']} )
rc( 'font', size=50 )
rc( 'text', usetex=True )
rc( 'pdf',  fonttype=42 )
rc( 'ps',   fonttype=42 )

x_max = 200
y_max = 2
x = np.linspace( 0, x_max, 500 )
y = np.linspace( 0, y_max, 500 )

X, Y = np.meshgrid( x, y )
Z = 1000 * np.exp( Y**2 ) / (1+X**2)

fig = plt.figure( figsize=[8,8], frameon=True, dpi=600 )
ax = plt.Axes( fig, [0., 0., 1., 1.] )
fig.add_axes( ax )

cNorm     = colors.Normalize( vmin=0, vmax=100 )
scalarMap = cmx.ScalarMappable( norm=cNorm, cmap='RdYlBu_r' )
levels = [.01, .1, 1] + [10*i for i in range(1,11)]

CS = plt.contour( X, Y, Z,
                  levels=levels,
                  cmap='RdYlBu_r',
                  norm=cNorm )
class nf( float ):
     def __repr__( self ):
         str = '%.1f' % ( self.__float__(), )
         if 10 < self.__float__() < 100:
            return ''
         elif str[-1]=='0':
             return '%.0f' % self.__float__()
         else:
             return '%.1f' % self.__float__()
CS.levels = [nf(val) for val in CS.levels ]
plt.clabel( CS, CS.levels, inline=1, fmt=r'%r', fontsize=40 )

# colorbar
scalarMap._A = []
cax  = fig.add_axes( [1.01, 0, 0.04, 1] )
cbar = fig.colorbar( scalarMap, ticks=[0, 50, 100], cax=cax )
cbar.ax.yaxis.set_ticks_position( 'right' )
cbar.ax.set_yticklabels( ['0', '50', '100' ] )

# axes
ax.tick_params( right='off', top='off' )
ax.set_xticks( [ 50, 100, 150, 200 ] )
ax.set_yticks( [ 0, 1.0, 2.0 ] )
ax.set_xlabel( r'$\Vert\mathbf{y}^s_i-\mathbf{y}^s_j\Vert$' )
ax.set_ylabel( r'$\Vert\mathbf{y}^t_i-\mathbf{y}^t_j\Vert$' )
ax.set_title(  r'$\exp(\Vert\mathbf{y}^t_i-\mathbf{y}^t_j\Vert^2)/(1+\Vert\mathbf{y}^s_i-\mathbf{y}^s_j\Vert^2)$', fontsize=40 )

fig.savefig( 'contour.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

