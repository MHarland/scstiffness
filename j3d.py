import unittest, os, numpy as np, h5py, itertools as itt
from pytriqs.gf.local import BlockGf, GfImFreq, GfImTime, inverse, iOmega_n
from pytriqs.utility import mpi
from bethe.h5interface import Storage
from bethe.setups.cdmftsquarelattice import NambuMomentumPlaquetteSetup as Setup
from periodization.dmft import LatticeGreensfunction

from mpiLists import scatter_list, allgather_list
from symmetry import symmetrize

    
class JosephsonExchange:
    def __init__(self, fname, nk, niw, tnn, tnnn, tz = -.15, rjs = [(0,0,0), (0,1,0), (0,0,1), (1,1,0),(1,0,1),(1,1,1),(2,0,0),(0,0,2)], verbose = True, loops = [-1]):
        self.verbose = verbose
        self.report('loading '+fname+'...')
        self.values = dict()
        sto = Storage(fname)
        seimpin = sto.load('se_imp_iw', -1)
        beta = seimpin.beta
        seimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = beta, n_points = niw, indices = b.indices)) for s, b in seimpin])
        seimptmp = seimp.copy()
        gimpin = sto.load('g_imp_iw', -1)
        gimp = BlockGf(name_block_generator = [(s, GfImFreq(beta = beta, n_points = niw, indices = b.indices)) for s, b in gimpin])
        gimptmp = gimp.copy()
        for loop in loops:
            seimpin = sto.load('se_imp_iw', loop)
            for s, b in seimpin:
                gtau = GfImTime(indices = b.indices, n_points = 10001, beta = beta)
                gtau.set_from_inverse_fourier(seimpin[s])
                seimptmp[s].set_from_fourier(gtau)
            seimp += seimptmp/len(loops)
            gimpin = sto.load('g_imp_iw', loop)
            for s, b in gimpin:
                gtau = GfImTime(indices = b.indices, n_points = 10001, beta = beta)
                gtau.set_from_inverse_fourier(gimpin[s])
                gimptmp[s].set_from_fourier(gtau)
            gimp += gimptmp/len(loops)
        
        seimp << symmetrize(seimp)
        gimp << symmetrize(gimp)
        
        mu = sto.load('mu', loop)
        setup = Setup(beta, mu, 999, tnn, tnnn, nk)

        seimp = setup.reblock_ksum.transform(seimp)
        gimp = setup.reblock_ksum.transform(gimp)

        blocknames = [k for k in seimp.indices]
        blockindices = [seimp[bn].indices for bn in blocknames]
        translations = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],[0,0,1],[0,0,-1]]
        t, s = tnn, tnnn
        h_r = [{'0':[[-mu,t,t,s],[t,-mu,s,t],[t,s,-mu,t],[s,t,t,-mu]]},
               {'0':[[0,t,0,s],[0,0,0,0],[0,s,0,t],[0,0,0,0]]},
               {'0':[[0,0,0,s],[0,0,0,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,t,s],[0,0,s,t],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,s,0],[0,0,0,0],[0,0,0,0]]},
               {'0':[[0,0,0,0],[t,0,s,0],[0,0,0,0],[s,0,t,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,0,0,0],[s,0,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[t,s,0,0],[s,t,0,0]]},
               {'0':[[0,0,0,0],[0,0,0,0],[0,s,0,0],[0,0,0,0]]},
               {'0':[[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]]},
               {'0':[[tz,0,0,0],[0,tz,0,0],[0,0,tz,0],[0,0,0,tz]]}]
        mu = 0
        # particle hole transform dispersion and mu
        p3 = np.array([[1,0],[0,-1]])
        # K transform dispersion and mu
        u = .5*np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])

        for i_h, h in enumerate(h_r):
            h_r[i_h] = {'0': np.kron(p3, u.dot(np.array(h['0'])).dot(u))}
        weights_r = [1] * len(translations)
        glat = LatticeGreensfunction(blocknames, blockindices, translations, h_r, nk, seimp, mu, weights_r, gk_on_the_fly = True)

        ri = (0,0,0)
        gtmp = BlockGf(name_block_generator = [[j, GfImFreq(beta = beta, n_points = niw, indices = [0])] for j in gimp.indices])
        j_r = np.zeros([len(rjs),3])
        s = '0'
        for i_rj, rj in enumerate(rjs):
            self.report(str(rj))
            gtmp.zero()
            gtmp00 = gtmp[s][0, 0]
            glatij = glat[ri,rj][s]
            glatji = glatij
            for i, j, k, l in itt.product(*[range(4)]*4):
                if j!=k and not((j,k) in [(1,5),(5,1),(2,6),(6,2)]): continue
                if l!=i and not((l,i) in [(1,5),(5,1),(2,6),(6,2)]): continue
                gtmp00 += glatij[0+i,0+j]*seimp[s][0+j,4+k]*glatji[4+k,4+l]*seimp[s][0+l,4+i]
                gtmp00 -= glatij[0+i,4+j]*seimp[s][0+j,4+k]*glatji[0+k,4+l]*seimp[s][0+l,4+i]
            self.values[rj] = gtmp.total_density().real

    def report(self, text):
        if mpi.is_master_node() and self.verbose:
            print text

    def report_detail(self, text):
        if mpi.is_master_node() and self.verbose > 1:
            print text
