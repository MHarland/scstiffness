from pytriqs.utility import mpi


def symmetrize(g, verbose = False):
    if mpi.is_master_node() and verbose:
        print 'before'
        for s, b in g:
            print s
            print b.data[20,:,:]
        print
    gtmp = g.copy()
    # up down
    for s in ['G','X','Y','M']:
        g[s][0,0] << (gtmp[s][0,0]-gtmp[s][1,1].conjugate()) * .5
        g[s][1,1] << (-1*gtmp[s][0,0].conjugate()+gtmp[s][1,1]) * .5
    # no anomalous
    for s in ['G','M']:
        g[s][0,1] << 0
        g[s][1,0] << 0
    # gap symmetry- spatial
    g['X'][0,1] << (gtmp['X'][0,1]+gtmp['X'][1,0]-gtmp['Y'][0,1]-gtmp['Y'][1,0]) * .25
    g['X'][1,0] << (gtmp['X'][0,1]+gtmp['X'][1,0]-gtmp['Y'][0,1]-gtmp['Y'][1,0]) * .25
    g['Y'][0,1] << (gtmp['Y'][0,1]+gtmp['Y'][1,0]-gtmp['X'][0,1]-gtmp['X'][1,0]) * .25
    g['Y'][1,0] << (gtmp['Y'][0,1]+gtmp['Y'][1,0]-gtmp['X'][0,1]-gtmp['X'][1,0]) * .25
    gtmp = g.copy()
    # XY-symmetry normal
    for i in range(2):
        g['X'][i,i] << (gtmp['X'][i,i]+gtmp['Y'][i,i]) * .5
        g['Y'][i,i] << (gtmp['X'][i,i]+gtmp['Y'][i,i]) * .5
    # gap symmetry - real
    for s in ['X','Y']:
        g[s][0,1] << (gtmp[s][0,1] + gtmp[s][0,1].conjugate()) * .5
        g[s][1,0] << (gtmp[s][1,0] + gtmp[s][1,0].conjugate()) * .5
    if mpi.is_master_node() and verbose:
        print 'after'
        for s, b in g:
            print s
            print b.data[20,:,:]
        print
    return g
